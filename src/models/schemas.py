"""
数据模型定义
"""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TranscriptionSegment(BaseModel):
    """转录片段"""
    start_time: float = Field(..., description="开始时间（秒）")
    end_time: float = Field(..., description="结束时间（秒）")
    text: str = Field(..., description="转录文本")
    speaker: str = Field(..., description="说话人标识")
    
    model_config = {"protected_namespaces": ()}


class TranscriptionResult(BaseModel):
    """转录结果"""
    task_id: str = Field(..., description="任务ID")
    file_name: str = Field(..., description="文件名")
    file_hash: str = Field(..., description="文件哈希值")
    duration: float = Field(..., description="音频时长（秒）")
    segments: List[TranscriptionSegment] = Field(..., description="转录片段列表")
    speakers: List[str] = Field(..., description="说话人列表")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    processing_time: float = Field(..., description="处理时长（秒）")
    error: Optional[str] = Field(None, description="错误信息")
    
    model_config = {"protected_namespaces": ()}


class TranscriptionTask(BaseModel):
    """转录任务"""
    task_id: str = Field(..., description="任务ID")
    file_name: str = Field(..., description="文件名")
    file_path: str = Field(..., description="文件路径")
    file_size: int = Field(..., description="文件大小（字节）")
    file_hash: str = Field(..., description="文件哈希值")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="任务状态")
    progress: float = Field(default=0.0, description="进度（0-100）")
    created_at: datetime = Field(default_factory=datetime.now, description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    retry_count: int = Field(default=0, description="重试次数")
    error: Optional[str] = Field(None, description="错误信息")
    result: Optional[TranscriptionResult] = Field(None, description="转录结果")
    force_refresh: bool = Field(default=False, description="强制刷新缓存")
    
    model_config = {"protected_namespaces": ()}


class WebSocketMessage(BaseModel):
    """WebSocket消息"""
    type: str = Field(..., description="消息类型")
    data: Dict[str, Any] = Field(..., description="消息数据")
    timestamp: datetime = Field(default_factory=datetime.now, description="时间戳")
    
    model_config = {"protected_namespaces": ()}


class AuthRequest(BaseModel):
    """认证请求"""
    username: str = Field(..., description="用户名")
    password: str = Field(..., description="密码")
    
    model_config = {"protected_namespaces": ()}


class AuthResponse(BaseModel):
    """认证响应"""
    access_token: str = Field(..., description="访问令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间（秒）")
    
    model_config = {"protected_namespaces": ()}


class FileUploadRequest(BaseModel):
    """文件上传请求"""
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小（字节）")
    file_hash: str = Field(..., description="文件哈希值")
    force_refresh: bool = Field(default=False, description="强制刷新缓存")
    
    model_config = {"protected_namespaces": ()}


class FileUploadResponse(BaseModel):
    """文件上传响应"""
    task_id: str = Field(..., description="任务ID")
    upload_url: Optional[str] = Field(None, description="上传URL")
    status: str = Field(..., description="状态")
    message: str = Field(..., description="消息")
    
    model_config = {"protected_namespaces": ()}


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(..., description="任务ID")
    status: TaskStatus = Field(..., description="任务状态")
    progress: float = Field(..., description="进度（0-100）")
    result: Optional[TranscriptionResult] = Field(None, description="转录结果")
    error: Optional[str] = Field(None, description="错误信息")
    
    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误信息")
    details: Optional[Dict[str, Any]] = Field(None, description="详细信息")
    
    model_config = {"protected_namespaces": ()}