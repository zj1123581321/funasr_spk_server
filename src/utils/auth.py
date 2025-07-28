"""
认证工具模块
"""
from datetime import datetime, timedelta
from typing import Optional, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from loguru import logger
from src.core.config import config


# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """获取密码哈希"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """创建访问令牌"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=config.auth.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, config.auth.secret_key, algorithm=config.auth.algorithm)
    
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict]:
    """验证令牌"""
    if not config.auth.enabled:
        return {"username": "anonymous"}
    
    try:
        payload = jwt.decode(token, config.auth.secret_key, algorithms=[config.auth.algorithm])
        username: str = payload.get("sub")
        if username is None:
            return None
        return {"username": username}
    except JWTError as e:
        logger.debug(f"令牌验证失败: {e}")
        return None


# 简单的用户存储（实际应用中应该使用数据库）
# 默认用户: admin/admin123
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": get_password_hash("admin123"),
        "disabled": False,
    }
}


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """认证用户"""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    if user.get("disabled"):
        return None
    return user