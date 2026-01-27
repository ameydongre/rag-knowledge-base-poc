import json
import boto3
import os
from datetime import datetime
import hashlib

s3_client = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
bedrock_runtime = boto3.client('bedrock-runtime', region_name='us-east-1')

DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'DocumentMetadata')
EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'

def lambda_handler(event, context):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        print(f"Processing document: {key} from bucket: {bucket}")
        
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_content = response['Body'].read()
        
        file_extension = key.split('.')[-1].lower()
        text_content = extract_text(file_content, file_extension)
        
        if not text_content:
            print(f"No text extracted from {key}")
            return {'statusCode': 200, 'body': 'No text content'}
        
        chunks = split_into_chunks(text_content, max_length=8000)
        document_id = hashlib.md5(key.encode()).hexdigest()
        table = dynamodb.Table(DYNAMODB_TABLE)
        
        for idx, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            chunk_id = f"{document_id}_chunk_{idx}"
            
            metadata = {
                'document_id': document_id,
                'chunk_id': chunk_id,
                'document_name': key.split('/')[-1],
                's3_bucket': bucket,
                's3_key': key,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'text_content': chunk[:1000],
                'full_text_s3_key': f"processed/{document_id}_chunk_{idx}.txt",
                'embedding_dimension': len(embedding) if embedding else 0,
                'document_type': file_extension,
                'created_date': datetime.utcnow().isoformat(),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            s3_client.put_object(
                Bucket=bucket,
                Key=metadata['full_text_s3_key'],
                Body=chunk.encode('utf-8')
            )
            
            table.put_item(Item=metadata)
            print(f"Processed chunk {idx + 1}/{len(chunks)} for document {key}")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Successfully processed {len(chunks)} chunks from {key}')
        }
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }

def extract_text(file_content, file_extension):
    try:
        if file_extension == 'txt':
            return file_content.decode('utf-8')
        
        elif file_extension == 'pdf':
            import PyPDF2
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif file_extension in ['doc', 'docx']:
            import docx
            import io
            doc = docx.Document(io.BytesIO(file_content))
            text = "".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        
        else:
            return file_content.decode('utf-8', errors='ignore')
            
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return ""

def split_into_chunks(text, max_length=8000):
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_embedding(text):
    try:
        body = json.dumps({"inputText": text})
        
        response = bedrock_runtime.invoke_model(
            modelId=EMBEDDING_MODEL_ID,
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])
        return embedding
        
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None



