import boto3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config as config

class S3Service:
    """
    A service class for interacting with Amazon S3.
    Provides methods to upload, download, list, and delete files in S3 buckets,
    as well as retrieve file metadata, content, and content type.
    """

    def __init__(self):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION
        )

    def upload_file(self, file_path, bucket_name, object_name):
        self.s3.upload_file(file_path, bucket_name, object_name)

    def download_file(self, bucket_name, object_name, file_path):
        self.s3.download_file(bucket_name, object_name, file_path)

    def list_files(self, bucket_name):
        response = self.s3.list_objects_v2(Bucket=bucket_name)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def delete_file(self, bucket_name, object_name):
        self.s3.delete_object(Bucket=bucket_name, Key=object_name)

    def upload_file_to_s3(self, file_path, bucket_name, object_name):
        self.s3.upload_file(file_path, bucket_name, object_name)

    def download_file_from_s3(self, bucket_name, object_name, file_path):
        self.s3.download_file(bucket_name, object_name, file_path)

    def list_files_in_bucket(self, bucket_name):
        response = self.s3.list_objects_v2(Bucket=bucket_name)
        return [obj['Key'] for obj in response.get('Contents', [])]

    def delete_file_from_s3(self, bucket_name, object_name):
        self.s3.delete_object(Bucket=bucket_name, Key=object_name)

    def get_file_metadata_from_s3(self, bucket_name, object_name):
        response = self.s3.head_object(Bucket=bucket_name, Key=object_name)
        return response['Metadata']

    def get_file_content_from_s3(self, bucket_name, object_name):
        response = self.s3.get_object(Bucket=bucket_name, Key=object_name)
        return response['Body'].read()

    def get_file_content_type_from_s3(self, bucket_name, object_name):
        response = self.s3.head_object(Bucket=bucket_name, Key=object_name)
        return response['ContentType']

if __name__ == "__main__":
    import os

    # Fix the path to be relative to the current script location
    test_txt_path = os.path.join(os.path.dirname(__file__), "test.txt")
    print(f"Location of test.txt: {test_txt_path}")

    s3_service = S3Service()
    s3_service.upload_file(test_txt_path, config.AWS_BUCKET_NAME, "test.txt")
    print(s3_service.list_files(config.AWS_BUCKET_NAME))
    s3_service.delete_file(config.AWS_BUCKET_NAME, "test.txt")
    print(s3_service.list_files(config.AWS_BUCKET_NAME))
