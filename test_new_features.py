"""
Test script to demonstrate the new framework, cloud, and container analysis features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from code_analyzer.framework_analyzer import FrameworkAnalyzer
from code_analyzer.cloud_analyzer import CloudAnalyzer
from code_analyzer.container_analyzer import ContainerAnalyzer

def test_framework_analyzer():
    """Test the framework analyzer with React and Django examples."""
    print("=== Testing Framework Analyzer ===")
    
    analyzer = FrameworkAnalyzer()
    
    # Test React code
    react_code = """
import React, { useState } from 'react';

function MyComponent() {
    const [data, setData] = useState(null);
    
    // Bad: API call without useEffect
    fetch('/api/data').then(res => setData(res.json()));
    
    return (
        <div>
            {data && data.map(item => (
                <div>{item.name}</div>  // Bad: missing key prop
            ))}
        </div>
    );
}
"""
    
    result = analyzer.analyze_code('MyComponent.jsx', react_code)
    print(f"React Analysis: {result['framework']}")
    print(f"Found {result['total_issues']} issues:")
    for issue in result['issues']:
        print(f"  - {issue['message']}")
    
    # Test Django code
    django_code = """
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .models import User

@csrf_exempt
def my_view(request):
    # Bad: Raw SQL query
    users = User.objects.raw("SELECT * FROM users WHERE active = 1")
    return render(request, 'users.html', {'users': users})
"""
    
    result = analyzer.analyze_code('views.py', django_code)
    print(f"\nDjango Analysis: {result['framework']}")
    print(f"Found {result['total_issues']} issues:")
    for issue in result['issues']:
        print(f"  - {issue['message']}")

def test_cloud_analyzer():
    """Test the cloud analyzer with AWS and Azure examples."""
    print("\n=== Testing Cloud Analyzer ===")
    
    analyzer = CloudAnalyzer()
    
    # Test AWS code
    aws_code = """
import boto3

# Bad: Hardcoded credentials
aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

s3 = boto3.client('s3')
# Bad: No error handling
s3.upload_file('myfile.txt', 'my-bucket', 'myfile.txt')
"""
    
    result = analyzer.analyze_code('aws_upload.py', aws_code)
    print(f"AWS Analysis: {result['platform']}")
    print(f"Found {result['total_issues']} issues:")
    for issue in result['issues']:
        print(f"  - {issue['message']}")
    
    # Test Azure code
    azure_code = """
from azure.storage.blob import BlobServiceClient

# Bad: Hardcoded connection string
connection_string = "DefaultEndpointsProtocol=https;AccountName=myaccount;AccountKey=mykey;EndpointSuffix=core.windows.net"

blob_service_client = BlobServiceClient.from_connection_string(connection_string)
"""
    
    result = analyzer.analyze_code('azure_storage.py', azure_code)
    print(f"\nAzure Analysis: {result['platform']}")
    print(f"Found {result['total_issues']} issues:")
    for issue in result['issues']:
        print(f"  - {issue['message']}")

def test_container_analyzer():
    """Test the container analyzer with Dockerfile and Kubernetes examples."""
    print("\n=== Testing Container Analyzer ===")
    
    analyzer = ContainerAnalyzer()
    
    # Test Dockerfile
    dockerfile_content = """
FROM python:latest
COPY . /app
RUN pip install -r requirements.txt
USER root
CMD ["python", "app.py"]
"""
    
    result = analyzer.analyze_code('Dockerfile', dockerfile_content)
    print(f"Dockerfile Analysis: {result['config_type']}")
    print(f"Found {result['total_issues']} issues:")
    for issue in result['issues']:
        print(f"  - {issue['message']}")
    
    # Test Kubernetes YAML
    k8s_content = """
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  containers:
  - name: app
    image: myapp:latest
    ports:
    - containerPort: 8080
"""
    
    result = analyzer.analyze_code('pod.yaml', k8s_content)
    print(f"\nKubernetes Analysis: {result['config_type']}")
    print(f"Found {result['total_issues']} issues:")
    for issue in result['issues']:
        print(f"  - {issue['message']}")

def test_integration():
    """Test integration with the main analyzer."""
    print("\n=== Testing Integration with Main Analyzer ===")
    
    try:
        from code_analyzer.main import CodeAnalyzer
        
        analyzer = CodeAnalyzer(enable_rag=False)
        
        # Test with React code
        react_code = """
import React, { useState } from 'react';

function MyComponent() {
    const [data, setData] = useState(null);
    fetch('/api/data').then(res => setData(res.json()));
    
    return (
        <div>
            {data && data.map(item => (
                <div>{item.name}</div>
            ))}
        </div>
    );
}
"""
        
        result = analyzer.analyze_code(
            code=react_code,
            model="deepseek",
            file_path="MyComponent.jsx",
            mode="thorough"
        )
        
        print(f"Integration Test - Quality Score: {result.code_quality_score}")
        print(f"Bugs found: {len(result.potential_bugs)}")
        print(f"Suggestions: {len(result.improvement_suggestions)}")
        
        # Show framework-specific suggestions
        framework_suggestions = [s for s in result.improvement_suggestions if s.startswith('Framework')]
        if framework_suggestions:
            print("\nFramework-specific suggestions:")
            for suggestion in framework_suggestions:
                print(f"  - {suggestion}")
        
    except Exception as e:
        print(f"Integration test failed: {e}")

if __name__ == "__main__":
    print("Testing New Analysis Features")
    print("=" * 50)
    
    test_framework_analyzer()
    test_cloud_analyzer()
    test_container_analyzer()
    test_integration()
    
    print("\n" + "=" * 50)
    print("All tests completed!") 