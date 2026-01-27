
import json
import boto3
import os

# Initialize AWS clients
bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', region_name=os.environ.get('AWS_REGION', 'us-east-1'))
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.environ.get('AWS_REGION', 'us-east-1'))

# Environment variables
KNOWLEDGE_BASE_ID = os.environ.get('KNOWLEDGE_BASE_ID')
MODEL_ID = 'anthropic.claude-3-sonnet-20240229-v1:0'

def lambda_handler(event, context):
    """
    Processes user queries using RAG with Bedrock Knowledge Base.
    Returns AI-generated response with source citations.
    """
    try:
        # Parse request body
        if isinstance(event.get('body'), str):
            body = json.loads(event.get('body', '{}'))
        else:
            body = event.get('body', {})
        
        user_query = body.get('query', '')
        
        if not user_query:
            return {
                'statusCode': 400,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({'error': 'Query is required'})
            }
        
        print(f"Processing query: {user_query}")
        
        # Retrieve relevant documents from Knowledge Base
        retrieve_response = bedrock_agent_runtime.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={
                'text': user_query
            },
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': 5
                }
            }
        )
        
        # Extract retrieved documents
        retrieved_docs = retrieve_response.get('retrievalResults', [])
        
        if not retrieved_docs:
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'answer': 'I could not find relevant information to answer your question.',
                    'sources': []
                })
            }
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for idx, doc in enumerate(retrieved_docs):
            doc_content = doc.get('content', {}).get('text', '')
            doc_location = doc.get('location', {})
            s3_location = doc_location.get('s3Location', {})
            
            context_parts.append(f"Document {idx + 1}:{doc_content}")
            
            sources.append({
                'document': s3_location.get('uri', 'Unknown'),
                'score': doc.get('score', 0),
                'excerpt': doc_content[:200] + '...' if len(doc_content) > 200 else doc_content
            })
        
        context = "".join(context_parts)
        
        # Create prompt for Claude
        prompt = f"""Human: You are a helpful AI assistant specializing in AWS services and best practices. Answer the following question based on the provided context. If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {user_query}

Provide a clear, concise answer and mention which documents you used. Format your response in a helpful way.

Assistant:"""
        
        # Generate response using Claude
        response = bedrock_runtime.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps({
                'anthropic_version': 'bedrock-2023-05-31',
                'max_tokens': 1000,
                'messages': [{
                    'role': 'user',
                    'content': prompt
                }]
            })
        )
        
        response_body = json.loads(response['body'].read())
        answer = response_body['content'][0]['text']
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'answer': answer,
                'sources': sources
            })
        }
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': 'Internal server error'})
        }

