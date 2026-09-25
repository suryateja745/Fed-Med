"""
FedMed Relational Database Persistence Layer.
Implements SQLAlchemy ORM models, session management, and database initialization.
Supports SQLite (zero-config default) and MySQL/PostgreSQL via DATABASE_URL.
"""

from datetime import datetime, timezone
import enum
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import get_backend_config
from federation.utils.logger import setup_logger

logger = setup_logger(name="FedMedDB")


class Base(DeclarativeBase):
    """Declarative base class for all FedMed database models."""
    pass


class UserRole(str, enum.Enum):
    """User authorization roles."""
    ADMIN = "ADMIN"
    HOSPITAL_STAFF = "HOSPITAL_STAFF"
    AUDITOR = "AUDITOR"


class NodeStatus(str, enum.Enum):
    """Hospital client node operational status."""
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    TRAINING = "TRAINING"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

class User(Base):
    """Registered platform user with credentials and institutional role."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(64), unique=True, nullable=False, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    email = Column(String(128), unique=True, nullable=False, index=True)
    password_hash = Column(String(256), nullable=False)
    salt = Column(String(64), nullable=False)
    role = Column(SQLEnum(UserRole), default=UserRole.HOSPITAL_STAFF, nullable=False)
    institution_name = Column(String(128), nullable=False)
    hospital_node_id = Column(String(64), nullable=True, index=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize user model to dictionary safe for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value if isinstance(self.role, UserRole) else str(self.role),
            "institution_name": self.institution_name,
            "hospital_node_id": self.hospital_node_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class HospitalNode(Base):
    """Institutional hospital client node registered in the federated cluster."""
    __tablename__ = "hospital_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String(64), unique=True, nullable=False, index=True)
    hospital_name = Column(String(128), unique=True, nullable=False)
    region = Column(String(64), nullable=False, default="Global")
    api_key = Column(String(128), unique=True, nullable=False)
    gpu_device = Column(String(128), default="NVIDIA RTX 4090")
    vram_gb = Column(Float, default=24.0)
    cpu_cores = Column(Integer, default=16)
    status = Column(SQLEnum(NodeStatus), default=NodeStatus.ONLINE, nullable=False)
    dataset_path = Column(String(256), default="./data/hospital_a")
    local_sample_count = Column(Integer, default=0)
    last_heartbeat = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    registered_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hospital node to dictionary for API responses."""
        return {
            "id": self.id,
            "node_id": self.node_id,
            "hospital_name": self.hospital_name,
            "region": self.region,
            "gpu_device": self.gpu_device,
            "vram_gb": self.vram_gb,
            "cpu_cores": self.cpu_cores,
            "status": self.status.value if isinstance(self.status, NodeStatus) else str(self.status),
            "dataset_path": self.dataset_path,
            "local_sample_count": self.local_sample_count,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
        }


class TrainingSessionRecord(Base):
    """Historical federated learning training round record."""
    __tablename__ = "training_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), unique=True, nullable=False, index=True)
    round_number = Column(Integer, nullable=False, index=True)
    status = Column(String(32), default="COMPLETED", nullable=False)
    participating_nodes = Column(Text, nullable=False, default="[]")  # JSON encoded list
    mean_dice = Column(Float, nullable=False, default=0.0)
    tc_dice = Column(Float, nullable=False, default=0.0)
    wt_dice = Column(Float, nullable=False, default=0.0)
    et_dice = Column(Float, nullable=False, default=0.0)
    loss = Column(Float, nullable=False, default=0.0)
    model_checkpoint_path = Column(String(256), nullable=True)
    checkpoint_sha256 = Column(String(64), nullable=True)
    checkpoint_hmac = Column(String(64), nullable=True)
    duration_seconds = Column(Float, default=0.0)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize training session record."""
        nodes = []
        try:
            nodes = json.loads(self.participating_nodes)
        except Exception:
            nodes = [self.participating_nodes]
        return {
            "id": self.id,
            "session_id": self.session_id,
            "round_number": self.round_number,
            "status": self.status,
            "participating_nodes": nodes,
            "mean_dice": round(self.mean_dice, 4),
            "tc_dice": round(self.tc_dice, 4),
            "wt_dice": round(self.wt_dice, 4),
            "et_dice": round(self.et_dice, 4),
            "loss": round(self.loss, 4),
            "model_checkpoint_path": self.model_checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_hmac": self.checkpoint_hmac,
            "duration_seconds": round(self.duration_seconds, 2),
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AuditLogEntry(Base):
    """Regulatory and compliance audit log entry."""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
        index=True,
    )
    actor_id = Column(String(64), nullable=False, index=True)
    actor_role = Column(String(32), nullable=False)
    action = Column(String(64), nullable=False, index=True)
    target_resource = Column(String(128), nullable=False)
    client_ip = Column(String(64), nullable=True)
    details = Column(Text, nullable=True)  # JSON encoded metadata
    is_tamper_flagged = Column(Boolean, default=False, nullable=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize audit log entry."""
        meta = {}
        if self.details:
            try:
                meta = json.loads(self.details)
            except Exception:
                meta = {"raw": self.details}
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "actor_id": self.actor_id,
            "actor_role": self.actor_role,
            "action": self.action,
            "target_resource": self.target_resource,
            "client_ip": self.client_ip,
            "details": meta,
            "is_tamper_flagged": self.is_tamper_flagged,
        }


# ---------------------------------------------------------------------------
# Database Engine & Session Factory
# ---------------------------------------------------------------------------

_engine = None
_SessionFactory = None


def get_db_engine(db_url: Optional[str] = None):
    """
    Factory creating or returning the configured SQLAlchemy engine.
    Supports SQLite and MySQL/PostgreSQL connection URLs.
    """
    global _engine
    if _engine is not None and db_url is None:
        return _engine

    cfg = get_backend_config()
    target_url = db_url or cfg.database_url

    connect_args = {}
    if target_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Ensure parent directory exists for SQLite file
        if "///" in target_url:
            path_part = target_url.split("///")[-1]
            if path_part and not path_part.startswith(":memory:"):
                Path(path_part).parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(target_url, connect_args=connect_args, pool_pre_ping=True)

        # Optimize SQLite for concurrency: enable WAL mode and foreign keys
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA synchronous=NORMAL")
            except Exception:
                pass
            finally:
                cursor.close()
    else:
        # MySQL or PostgreSQL connection
        engine = create_engine(
            target_url,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )

    if db_url is None:
        _engine = engine
    return engine


def get_session_factory(engine=None) -> sessionmaker:
    """Return sessionmaker bound to the engine."""
    global _SessionFactory
    eng = engine or get_db_engine()
    if _SessionFactory is None or engine is not None:
        _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    return _SessionFactory


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a SQLAlchemy session.
    Commits on success, rolls back on exception, and guarantees session close.
    """
    factory = get_session_factory()
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Default Seeding & Initialization
# ---------------------------------------------------------------------------

def seed_default_data(db: Session) -> None:
    """
    Seeds default administrative users, hospital nodes, and auditor accounts
    if they do not already exist in the database.
    """
    from backend.security import hash_password

    # 1. Seed Central Coordinator Admin
    admin_user = db.query(User).filter(User.username == "admin").first()
    if not admin_user:
        p_hash, p_salt = hash_password("Admin@FedMed2026!")
        admin = User(
            user_id="USER-COORD-001",
            username="admin",
            email="coordinator@fedmed.ai",
            password_hash=p_hash,
            salt=p_salt,
            role=UserRole.ADMIN,
            institution_name="FedMed Central Command & AI Consortium",
            hospital_node_id=None,
            is_active=True,
        )
        db.add(admin)
        logger.info("  * Seeded default Coordinator Admin user (admin)")

    # 2. Seed Hospital Nodes (A, B, C)
    default_nodes = [
        {
            "node_id": "NODE-HOSP-A",
            "hospital_name": "Mount Sinai Brain Tumor Center",
            "region": "New York, USA",
            "api_key": "fedmed_live_key_hosp_a_89324789",
            "gpu_device": "NVIDIA RTX 4090 (24GB)",
            "vram_gb": 24.0,
            "cpu_cores": 16,
            "dataset_path": "./data/hospital_a",
            "local_sample_count": 48,
            "staff_username": "hosp_a_lead",
            "staff_email": "neuro.lead@mountsinai.org",
            "staff_pass": "HospA@FedMed2026!",
        },
        {
            "node_id": "NODE-HOSP-B",
            "hospital_name": "Johns Hopkins Neuro-Oncology Unit",
            "region": "Baltimore, USA",
            "api_key": "fedmed_live_key_hosp_b_32948234",
            "gpu_device": "NVIDIA A100 Tensor Core (80GB)",
            "vram_gb": 80.0,
            "cpu_cores": 32,
            "dataset_path": "./data/hospital_b",
            "local_sample_count": 64,
            "staff_username": "hosp_b_lead",
            "staff_email": "lead.fl@jhmi.edu",
            "staff_pass": "HospB@FedMed2026!",
        },
        {
            "node_id": "NODE-HOSP-C",
            "hospital_name": "Mayo Clinic Imaging Research Consortium",
            "region": "Rochester, USA",
            "api_key": "fedmed_live_key_hosp_c_19283746",
            "gpu_device": "NVIDIA RTX 3090 (24GB)",
            "vram_gb": 24.0,
            "cpu_cores": 16,
            "dataset_path": "./data/hospital_c",
            "local_sample_count": 36,
            "staff_username": "hosp_c_lead",
            "staff_email": "mri.research@mayo.edu",
            "staff_pass": "HospC@FedMed2026!",
        },
    ]

    for item in default_nodes:
        node = db.query(HospitalNode).filter(HospitalNode.node_id == item["node_id"]).first()
        if not node:
            new_node = HospitalNode(
                node_id=item["node_id"],
                hospital_name=item["hospital_name"],
                region=item["region"],
                api_key=item["api_key"],
                gpu_device=item["gpu_device"],
                vram_gb=item["vram_gb"],
                cpu_cores=item["cpu_cores"],
                status=NodeStatus.ONLINE,
                dataset_path=item["dataset_path"],
                local_sample_count=item["local_sample_count"],
            )
            db.add(new_node)
            logger.info(f"  * Seeded Hospital Node: {item['node_id']} ({item['hospital_name']})")

        staff = db.query(User).filter(User.username == item["staff_username"]).first()
        if not staff:
            p_hash, p_salt = hash_password(item["staff_pass"])
            new_staff = User(
                user_id=f"USER-{item['node_id'].replace('NODE-', '')}",
                username=item["staff_username"],
                email=item["staff_email"],
                password_hash=p_hash,
                salt=p_salt,
                role=UserRole.HOSPITAL_STAFF,
                institution_name=item["hospital_name"],
                hospital_node_id=item["node_id"],
                is_active=True,
            )
            db.add(new_staff)
            logger.info(f"  * Seeded Hospital Staff user ({item['staff_username']})")

    # 3. Seed Clinical Quality Auditor
    auditor_user = db.query(User).filter(User.username == "auditor").first()
    if not auditor_user:
        p_hash, p_salt = hash_password("Audit@FedMed2026!")
        auditor = User(
            user_id="USER-AUDIT-001",
            username="auditor",
            email="regulatory.audit@fedmed.ai",
            password_hash=p_hash,
            salt=p_salt,
            role=UserRole.AUDITOR,
            institution_name="Global Clinical Ethics & AI Governance Board",
            hospital_node_id=None,
            is_active=True,
        )
        db.add(auditor)
        logger.info("  * Seeded Clinical Auditor user (auditor)")

    db.commit()


def init_db(engine_override=None) -> None:
    """
    Initializes database tables and seeds standard accounts and hospital nodes.
    Safe to call repeatedly on server boot.
    """
    eng = engine_override or get_db_engine()
    logger.info(f"Initializing FedMed database schemas at: {eng.url}")
    Base.metadata.create_all(bind=eng)

    factory = get_session_factory(eng)
    with factory() as session:
        seed_default_data(session)
    logger.info("Database initialization and seed verification complete.")
