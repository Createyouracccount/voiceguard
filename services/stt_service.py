import logging
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import grpc
import pyaudio
from requests import Session

# 프로젝트 루트를 기준으로 pb2 파일들을 import 하도록 경로 설정
try:
    from . import vito_stt_client_pb2 as pb
    from . import vito_stt_client_pb2_grpc as pb_grpc
except ImportError:
    # 다른 환경에서도 동작하도록 예외 처리
    sys.path.append(str(Path(__file__).parent))
    import vito_stt_client_pb2 as pb
    import vito_stt_client_pb2_grpc as pb_grpc


logger = logging.getLogger(__name__)

# --- 설정 상수 ---
API_BASE = "https://openapi.vito.ai"
GRPC_SERVER_URL = "grpc-openapi.vito.ai:443"
SAMPLE_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = int(SAMPLE_RATE / 10)  # 100ms
ENCODING = pb.DecoderConfig.AudioEncoding.LINEAR16

def get_config(keywords=None):
    """동적 디코더 설정 생성"""
    return pb.DecoderConfig(
        sample_rate=SAMPLE_RATE,
        encoding=ENCODING,
        use_itn=True,
        use_disfluency_filter=False,
        use_profanity_filter=False,
        keywords=keywords or [],
    )

class MicrophoneStream:
    """마이크 입력을 처리하는 스트림 클래스"""
    def __init__(self, rate: int, chunk: int, channels: int, format_type):
        self._rate = rate
        self._chunk = chunk
        self._channels = channels
        self._format = format_type
        self._buff = queue.Queue()
        self.closed = True
        self._audio_interface = None
        self._audio_stream = None

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=self._format,
            channels=self._channels,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.closed:
            self.terminate()

    def terminate(self):
        if self._audio_stream and self._audio_stream.is_active():
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        if self._audio_interface:
            self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            yield chunk

class RTZROpenAPIClient:
    """ReturnZero API 인증 및 gRPC 통신 클라이언트"""
    def __init__(self, client_id: str, client_secret: str, transcript_callback: Callable):
        self.client_id = client_id
        self.client_secret = client_secret
        self.transcript_callback = transcript_callback
        self._sess = Session()
        self._token = None
        self.stream = None
        self.is_running = False

    @property
    def token(self):
        if self._token is None or self._token.get("expire_at", 0) < time.time():
            try:
                resp = self._sess.post(
                    f"{API_BASE}/v1/authenticate",
                    data={"client_id": self.client_id, "client_secret": self.client_secret},
                    timeout=5
                )
                resp.raise_for_status()
                self._token = resp.json()
            except Exception as e:
                logger.critical(f"VITO API 인증 실패: {e}")
                raise
        return self._token["access_token"]

    def transcribe_streaming_grpc(self):
        """gRPC를 이용한 스트리밍 STT 수행"""
        self.is_running = True
        try:
            with grpc.secure_channel(GRPC_SERVER_URL, credentials=grpc.ssl_channel_credentials()) as channel:
                stub = pb_grpc.OnlineDecoderStub(channel)
                cred = grpc.access_token_call_credentials(self.token)

                def req_iterator():
                    yield pb.DecoderRequest(streaming_config=get_config())
                    with MicrophoneStream(SAMPLE_RATE, CHUNK, CHANNELS, FORMAT) as stream:
                        self.stream = stream
                        for chunk in stream.generator():
                            if not self.is_running or chunk is None:
                                break
                            yield pb.DecoderRequest(audio_content=chunk)
                
                while self.is_running:
                    try:
                        resp_iter = stub.Decode(req_iterator(), credentials=cred)
                        logger.info("gRPC 스트림 시작됨.")
                        for resp in resp_iter:
                            if not self.is_running: break
                            for res in resp.results:
                                if res.is_final and res.alternatives:
                                    self.transcript_callback(res.alternatives[0].text)
                    except grpc.RpcError as e:
                         if self.is_running:
                            logger.error(f"gRPC 오류 발생: {e.code()} - {e.details()}. 5초 후 재시도합니다.")
                            time.sleep(5) # 재연결 시도
                    except Exception as e:
                        if self.is_running:
                            logger.error(f"STT 스트림 처리 중 예외 발생: {e}")
                        break
        finally:
            logger.info("gRPC 스트림 종료됨.")
            if self.stream:
                self.stream.terminate()
                
    def stop(self):
        """스트리밍 종료"""
        self.is_running = False
        if self.stream:
            self.stream.terminate()

class SttService:
    """
    STT 서비스를 관리하는 클래스 (음성 인식 시작/종료 제어)
    """
    def __init__(self, client_id: str, client_secret: str, transcript_callback: Callable):
        self.client = RTZROpenAPIClient(client_id, client_secret, transcript_callback)
        self.thread = None
        self.is_running = False

    def start(self):
        """STT 서비스를 백그라운드 스레드에서 시작"""
        if self.is_running:
            logger.warning("STT 서비스가 이미 실행 중입니다.")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self.client.transcribe_streaming_grpc, daemon=True)
        self.thread.start()
        logger.info("STT 서비스 시작됨.")

    def stop(self):
        """STT 서비스 종료"""
        if not self.is_running:
            return
            
        self.is_running = False
        self.client.stop()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        logger.info("STT 서비스 종료됨.")