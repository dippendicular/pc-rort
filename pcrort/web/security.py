from datetime import datetime, timedelta, timezone
import secrets, jwt
from fastapi import HTTPException, Request
from passlib.context import CryptContext
from .config import settings

passwords = CryptContext(schemes=["bcrypt"], deprecated="auto")
def hash_password(value: str) -> str: return passwords.hash(value)
def verify_password(value: str, digest: str) -> bool: return passwords.verify(value, digest)
def issue_token(user_id: str, admin: bool) -> str:
    return jwt.encode({"sub":user_id,"admin":admin,"exp":datetime.now(timezone.utc)+timedelta(minutes=20)}, settings.jwt_secret, algorithm="HS256")
def decode_token(token: str) -> dict:
    try: return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except jwt.PyJWTError as exc: raise HTTPException(401, "Invalid or expired session") from exc
def csrf() -> str: return secrets.token_urlsafe(32)
def require_csrf(request: Request):
    if request.method in {"POST","PUT","PATCH","DELETE"} and request.headers.get("X-CSRF-Token") != request.cookies.get("csrf"):
        raise HTTPException(403, "CSRF validation failed")
