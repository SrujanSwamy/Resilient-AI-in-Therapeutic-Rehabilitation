"""
Publisher-Subscriber Communication Model
-----------------------------------------
End-Semester Enhancement: Decoupled message-based architecture between
Edge and Cloud components.

Design (PPT Slide 10):
    Edge Client  --publish-->  Message Broker  --subscribe-->  Cloud Server

Two transport modes:
    1. "local"  — thread-safe in-process queue (no dependencies)
    2. "zmq"    — ZeroMQ sockets (requires pyzmq) for real network comms

The API is identical for both modes so the rest of the system is
transport-agnostic.
"""

import json
import queue
import threading
import logging
from typing import Callable, Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Message format
# -----------------------------------------------------------------------

def build_message(topic: str, payload: Dict, sender: str = "edge") -> Dict:
    """Create a standardised broker message."""
    return {
        "topic": topic,
        "sender": sender,
        "timestamp": datetime.now().isoformat(),
        "payload": payload,
    }


# -----------------------------------------------------------------------
# Core Broker (local, in-process)
# -----------------------------------------------------------------------

class MessageBroker:
    """
    Thread-safe publish/subscribe message broker.

    Supports multiple topics and multiple subscribers per topic.
    Works 100% in-process via Python queues — no external services needed.

    Parameters
    ----------
    mode : str  "local" | "zmq"
        "zmq" requires pyzmq and network addresses.
    zmq_pub_addr : str
        ZeroMQ publisher bind address (e.g. "tcp://*:5555").
    zmq_sub_addr : str
        ZeroMQ subscriber connect address (e.g. "tcp://localhost:5555").
    """

    def __init__(
        self,
        mode: str = "local",
        zmq_pub_addr: str = "tcp://*:5555",
        zmq_sub_addr: str = "tcp://localhost:5555",
    ):
        self.mode = mode
        self._subscribers: Dict[str, List[Callable]] = {}
        self._queue: queue.Queue = queue.Queue()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # ZeroMQ setup (optional)
        self._zmq_pub = None
        self._zmq_sub = None
        self._zmq_context = None

        if mode == "zmq":
            self._init_zmq(zmq_pub_addr, zmq_sub_addr)

    def _init_zmq(self, pub_addr: str, sub_addr: str):
        try:
            import zmq
            self._zmq_context = zmq.Context()
            self._zmq_pub = self._zmq_context.socket(zmq.PUB)
            self._zmq_pub.bind(pub_addr)
            self._zmq_sub = self._zmq_context.socket(zmq.SUB)
            self._zmq_sub.connect(sub_addr)
            self._zmq_sub.setsockopt_string(zmq.SUBSCRIBE, "")
            logger.info(f"ZeroMQ broker initialised: pub={pub_addr}, sub={sub_addr}")
        except ImportError:
            logger.warning("pyzmq not installed — falling back to local mode.")
            self.mode = "local"
        except Exception as e:
            logger.warning(f"ZeroMQ init failed ({e}) — falling back to local mode.")
            self.mode = "local"

    # ------------------------------------------------------------------
    # Pub/Sub API
    # ------------------------------------------------------------------

    def subscribe(self, topic: str, handler: Callable[[Dict], None]):
        """Register a handler for a topic. Handler receives the full message dict."""
        self._subscribers.setdefault(topic, [])
        self._subscribers[topic].append(handler)
        logger.debug(f"Subscribed to topic '{topic}': {handler.__name__}")

    def publish(self, topic: str, payload: Dict, sender: str = "edge"):
        """Publish a message to a topic."""
        message = build_message(topic, payload, sender)

        if self.mode == "zmq" and self._zmq_pub:
            try:
                self._zmq_pub.send_string(json.dumps(message))
                return
            except Exception as e:
                logger.warning(f"ZMQ publish failed: {e}")

        # Local: put in queue
        self._queue.put(message)

    def process_one(self):
        """Process a single message from the queue (blocking with timeout)."""
        try:
            message = self._queue.get(timeout=0.1)
            topic = message.get("topic", "")
            handlers = self._subscribers.get(topic, []) + self._subscribers.get("*", [])
            for handler in handlers:
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Handler error on topic '{topic}': {e}")
        except queue.Empty:
            pass

    def start(self):
        """Start background message processing thread."""
        self._running = True
        self._worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._worker_thread.start()
        logger.info("MessageBroker started.")

    def stop(self):
        """Stop background thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2)
        if self._zmq_context:
            self._zmq_context.term()
        logger.info("MessageBroker stopped.")

    def _run_loop(self):
        while self._running:
            self.process_one()

    def flush(self, timeout: float = 2.0):
        """Process all pending messages (useful in tests)."""
        import time
        deadline = time.time() + timeout
        while not self._queue.empty() and time.time() < deadline:
            self.process_one()


# -----------------------------------------------------------------------
# Edge Publisher
# -----------------------------------------------------------------------

class EdgePublisher:
    """
    Publishes exercise session data from the Edge to the broker.

    Topics published:
        "session.result"   — full classification + enhancement result
        "session.alert"    — compensation or low-confidence alert
        "heartbeat"        — periodic liveness signal
    """

    def __init__(self, broker: MessageBroker, edge_id: str = "edge_01"):
        self.broker = broker
        self.edge_id = edge_id

    def publish_session(self, session_data: Dict) -> str:
        """
        Publish a completed session result.

        Parameters
        ----------
        session_data : dict
            Must include: patient_id, exercise_id, correctness, confidence,
            rmse, compensation_found, fluidity_score, etc.

        Returns
        -------
        str : message timestamp
        """
        self.broker.publish("session.result", session_data, sender=self.edge_id)
        msg_time = datetime.now().isoformat()

        # Additionally publish alert if compensation found or low confidence
        if session_data.get("compensation_found"):
            alert = {
                "patient_id": session_data.get("patient_id"),
                "exercise_id": session_data.get("exercise_id"),
                "alert_type": "compensation",
                "details": session_data.get("compensation_types", []),
            }
            self.broker.publish("session.alert", alert, sender=self.edge_id)

        if session_data.get("confidence", 1.0) < 0.8:
            alert = {
                "patient_id": session_data.get("patient_id"),
                "exercise_id": session_data.get("exercise_id"),
                "alert_type": "low_confidence",
                "confidence": session_data.get("confidence"),
            }
            self.broker.publish("session.alert", alert, sender=self.edge_id)

        return msg_time

    def publish_heartbeat(self):
        self.broker.publish(
            "heartbeat",
            {"edge_id": self.edge_id, "status": "alive"},
            sender=self.edge_id,
        )


# -----------------------------------------------------------------------
# Cloud Subscriber
# -----------------------------------------------------------------------

class CloudSubscriber:
    """
    Receives messages from the broker and stores them in the database.

    Parameters
    ----------
    broker    : MessageBroker
    database  : RehabDatabase (from subteam2_cloud.database)
    """

    def __init__(self, broker: MessageBroker, database=None):
        self.broker = broker
        self.database = database
        self._received: List[Dict] = []
        self._alerts: List[Dict] = []

        # Register handlers
        broker.subscribe("session.result", self.on_session_received)
        broker.subscribe("session.alert", self.on_alert_received)
        broker.subscribe("heartbeat", self.on_heartbeat)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def on_session_received(self, message: Dict):
        """Handler for 'session.result' topic."""
        payload = message.get("payload", {})
        self._received.append(payload)
        logger.info(
            f"[Cloud] Session received from {message.get('sender')}: "
            f"patient={payload.get('patient_id')}, "
            f"exercise={payload.get('exercise_id')}, "
            f"result={'Correct' if payload.get('correctness') == 1 else 'Incorrect'}"
        )
        if self.database is not None:
            try:
                session_id = self.database.save_session(payload.copy())
                logger.info(f"[Cloud] Saved to DB: session_id={session_id}")
            except Exception as e:
                logger.error(f"[Cloud] DB save error: {e}")

    def on_alert_received(self, message: Dict):
        """Handler for 'session.alert' topic."""
        payload = message.get("payload", {})
        self._alerts.append(payload)
        logger.warning(
            f"[Cloud] ALERT from {message.get('sender')}: "
            f"type={payload.get('alert_type')}, "
            f"patient={payload.get('patient_id')}"
        )

    def on_heartbeat(self, message: Dict):
        payload = message.get("payload", {})
        logger.debug(f"[Cloud] Heartbeat from {payload.get('edge_id')}")

    @property
    def received_sessions(self) -> List[Dict]:
        return list(self._received)

    @property
    def received_alerts(self) -> List[Dict]:
        return list(self._alerts)


# -----------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)

    print("Testing Publisher-Subscriber Module...")

    broker = MessageBroker(mode="local")
    cloud_sub = CloudSubscriber(broker)
    edge_pub = EdgePublisher(broker, edge_id="edge_01")

    broker.start()

    # Simulate session results from edge
    test_sessions = [
        {
            'patient_id': 'patient_001',
            'exercise_id': 'Ex1',
            'correctness': 1,
            'confidence': 0.92,
            'rmse': 0.21,
            'compensation_found': False,
            'compensation_types': [],
            'fluidity_score': 0.78,
            'fluidity_interpretation': 'Good',
        },
        {
            'patient_id': 'patient_002',
            'exercise_id': 'Ex2',
            'correctness': 0,
            'confidence': 0.65,
            'rmse': 0.48,
            'compensation_found': True,
            'compensation_types': ['trunk_lean'],
            'fluidity_score': 0.41,
            'fluidity_interpretation': 'Fair',
        },
    ]

    for session in test_sessions:
        edge_pub.publish_session(session)

    edge_pub.publish_heartbeat()
    broker.flush(timeout=2)

    print(f"\n  Sessions received by cloud: {len(cloud_sub.received_sessions)}")
    print(f"  Alerts received by cloud:   {len(cloud_sub.received_alerts)}")
    broker.stop()
    print("  [OK] Pub/Sub module OK")
