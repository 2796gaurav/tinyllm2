"""
S3 Utilities for SageMaker Training
Handles uploading/downloading models, checkpoints, and results to/from S3
"""

import os
import boto3
import logging
from pathlib import Path
from typing import Optional
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_s3_client():
    """Get S3 client (uses SageMaker role or default credentials)"""
    return boto3.client('s3')


def parse_s3_path(s3_path: str) -> tuple:
    """
    Parse S3 path into bucket and key
    
    Args:
        s3_path: S3 path like 's3://bucket/path/to/file'
    
    Returns:
        (bucket, key) tuple
    """
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'")
    
    s3_path = s3_path[5:]  # Remove 's3://'
    parts = s3_path.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    return bucket, key


def upload_to_s3(local_path: str, s3_path: str):
    """
    Upload a file to S3
    
    Args:
        local_path: Local file path
        s3_path: S3 path (e.g., 's3://bucket/path/to/file')
    """
    s3_client = get_s3_client()
    bucket, key = parse_s3_path(s3_path)
    
    try:
        logger.info(f"Uploading {local_path} to {s3_path}...")
        s3_client.upload_file(local_path, bucket, key)
        logger.info(f"✅ Uploaded to {s3_path}")
    except ClientError as e:
        logger.error(f"❌ Failed to upload {local_path} to {s3_path}: {e}")
        raise


def download_from_s3(s3_path: str, local_path: str):
    """
    Download a file from S3
    
    Args:
        s3_path: S3 path (e.g., 's3://bucket/path/to/file')
        local_path: Local file path to save to
    """
    s3_client = get_s3_client()
    bucket, key = parse_s3_path(s3_path)
    
    # Create parent directory if needed
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading {s3_path} to {local_path}...")
        s3_client.download_file(bucket, key, local_path)
        logger.info(f"✅ Downloaded to {local_path}")
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.error(f"❌ File not found: {s3_path}")
        else:
            logger.error(f"❌ Failed to download {s3_path}: {e}")
        raise


def sync_directory_to_s3(local_dir: str, s3_path: str):
    """
    Sync a local directory to S3 (recursive upload)
    
    Args:
        local_dir: Local directory path
        s3_path: S3 path (e.g., 's3://bucket/path/to/dir')
    """
    s3_client = get_s3_client()
    bucket, s3_prefix = parse_s3_path(s3_path)
    
    local_path = Path(local_dir)
    if not local_path.exists():
        logger.warning(f"Local directory does not exist: {local_dir}")
        return
    
    logger.info(f"Syncing {local_dir} to {s3_path}...")
    
    uploaded = 0
    for file_path in local_path.rglob('*'):
        if file_path.is_file():
            # Get relative path from local_dir
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{relative_path}".replace('\\', '/')
            
            try:
                s3_client.upload_file(str(file_path), bucket, s3_key)
                uploaded += 1
            except ClientError as e:
                logger.error(f"Failed to upload {file_path}: {e}")
    
    logger.info(f"✅ Synced {uploaded} files to {s3_path}")


def sync_directory_from_s3(s3_path: str, local_dir: str):
    """
    Sync a directory from S3 to local (recursive download)
    
    Args:
        s3_path: S3 path (e.g., 's3://bucket/path/to/dir')
        local_dir: Local directory path
    """
    s3_client = get_s3_client()
    bucket, s3_prefix = parse_s3_path(s3_path)
    
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Syncing {s3_path} to {local_dir}...")
    
    try:
        # List all objects with the prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
        
        downloaded = 0
        for page in pages:
            if 'Contents' not in page:
                continue
            
            for obj in page['Contents']:
                s3_key = obj['Key']
                
                # Skip if it's a directory marker
                if s3_key.endswith('/'):
                    continue
                
                # Get relative path
                if s3_key.startswith(s3_prefix):
                    relative_path = s3_key[len(s3_prefix):].lstrip('/')
                else:
                    relative_path = s3_key.split('/')[-1]
                
                local_file_path = local_path / relative_path
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download file
                s3_client.download_file(bucket, s3_key, str(local_file_path))
                downloaded += 1
        
        logger.info(f"✅ Synced {downloaded} files from {s3_path}")
    
    except ClientError as e:
        logger.error(f"❌ Failed to sync from {s3_path}: {e}")
        raise


def setup_s3_paths(s3_base_path: str):
    """
    Setup and verify S3 paths
    
    Args:
        s3_base_path: Base S3 path (e.g., 's3://bucket/path')
    """
    if not s3_base_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_base_path}. Must start with 's3://'")
    
    logger.info(f"S3 Base Path: {s3_base_path}")
    
    # Verify S3 access
    try:
        s3_client = get_s3_client()
        bucket, _ = parse_s3_path(s3_base_path)
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"✅ S3 access verified for bucket: {bucket}")
    except ClientError as e:
        logger.error(f"❌ Failed to access S3 bucket: {e}")
        raise
    
    return s3_base_path


def list_s3_files(s3_path: str) -> list:
    """
    List all files in an S3 path
    
    Args:
        s3_path: S3 path (e.g., 's3://bucket/path/to/dir')
    
    Returns:
        List of S3 keys
    """
    s3_client = get_s3_client()
    bucket, s3_prefix = parse_s3_path(s3_path)
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
        
        files = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if not obj['Key'].endswith('/'):
                        files.append(obj['Key'])
        
        return files
    except ClientError as e:
        logger.error(f"❌ Failed to list files in {s3_path}: {e}")
        raise


def check_s3_file_exists(s3_path: str) -> bool:
    """
    Check if a file exists in S3
    
    Args:
        s3_path: S3 path (e.g., 's3://bucket/path/to/file')
    
    Returns:
        True if file exists, False otherwise
    """
    s3_client = get_s3_client()
    bucket, key = parse_s3_path(s3_path)
    
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


