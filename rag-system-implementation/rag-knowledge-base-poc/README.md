# AWS RAG Knowledge Base POC

A complete Retrieval Augmented Generation (RAG) system built with Amazon Bedrock, Lambda, and API Gateway.

## Architecture

- **Document Processing**: Lambda function triggered by S3 uploads
- **Vector Storage**: Amazon Bedrock Knowledge Base with OpenSearch Serverless
- **Query Processing**: Lambda function with RAG capabilities
- **API**: REST API via API Gateway
- **Web Interface**: HTML served through API Gateway

## Prerequisites

- AWS Account
- AWS CLI configured
- Python 3.12
- Basic knowledge of AWS services

## Setup Instructions

See docs/SETUP.md for detailed deployment instructions.

## Cost Estimate

Approximately $110-185/month for light testing.

## License

MIT License
