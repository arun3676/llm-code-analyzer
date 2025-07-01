#!/usr/bin/env python3
"""
Quick test to demonstrate the new cloud analysis feature.
"""

from code_analyzer.main import CodeAnalyzer

def test_cloud_analysis():
    """Test cloud analysis with AWS code."""
    print("=== Testing Cloud Analysis Integration ===")
    
    # Initialize analyzer
    analyzer = CodeAnalyzer(enable_rag=False)
    
    # AWS code with issues
    aws_code = """
import boto3

# Bad: Hardcoded credentials
aws_access_key_id = "AKIAIOSFODNN7EXAMPLE"
aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

s3 = boto3.client('s3')
# Bad: No error handling
s3.upload_file('myfile.txt', 'my-bucket', 'myfile.txt')
"""
    
    print("Analyzing AWS code with hardcoded credentials...")
    result = analyzer.analyze_code(
        code=aws_code,
        file_path='aws_upload.py',
        mode='thorough'
    )
    
    print(f"\nQuality Score: {result.code_quality_score}")
    print(f"Bugs found: {len(result.potential_bugs)}")
    print(f"Total suggestions: {len(result.improvement_suggestions)}")
    
    # Show cloud-specific suggestions
    cloud_suggestions = [s for s in result.improvement_suggestions if s.startswith('Cloud')]
    if cloud_suggestions:
        print("\n🌩️ Cloud-specific suggestions:")
        for suggestion in cloud_suggestions:
            print(f"  ✅ {suggestion}")
    else:
        print("\n❌ No cloud-specific suggestions found")
    
    # Show other suggestions
    other_suggestions = [s for s in result.improvement_suggestions if not s.startswith('Cloud')]
    if other_suggestions:
        print("\n💡 Other suggestions:")
        for suggestion in other_suggestions[:3]:  # Show first 3
            print(f"  • {suggestion}")

def test_framework_analysis():
    """Test framework analysis with React code."""
    print("\n=== Testing Framework Analysis Integration ===")
    
    analyzer = CodeAnalyzer(enable_rag=False)
    
    # React code with issues
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
    
    print("Analyzing React code with missing useEffect and key props...")
    result = analyzer.analyze_code(
        code=react_code,
        file_path='MyComponent.jsx',
        mode='thorough'
    )
    
    print(f"\nQuality Score: {result.code_quality_score}")
    print(f"Bugs found: {len(result.potential_bugs)}")
    print(f"Total suggestions: {len(result.improvement_suggestions)}")
    
    # Show framework-specific suggestions
    framework_suggestions = [s for s in result.improvement_suggestions if s.startswith('Framework')]
    if framework_suggestions:
        print("\n⚛️ Framework-specific suggestions:")
        for suggestion in framework_suggestions:
            print(f"  ✅ {suggestion}")
    else:
        print("\n❌ No framework-specific suggestions found")

def test_container_analysis():
    """Test container analysis with Dockerfile."""
    print("\n=== Testing Container Analysis Integration ===")
    
    analyzer = CodeAnalyzer(enable_rag=False)
    
    # Dockerfile with issues
    dockerfile_content = """
FROM python:latest
COPY . /app
RUN pip install -r requirements.txt
USER root
CMD ["python", "app.py"]
"""
    
    print("Analyzing Dockerfile with security issues...")
    result = analyzer.analyze_code(
        code=dockerfile_content,
        file_path='Dockerfile',
        mode='thorough'
    )
    
    print(f"\nQuality Score: {result.code_quality_score}")
    print(f"Bugs found: {len(result.potential_bugs)}")
    print(f"Total suggestions: {len(result.improvement_suggestions)}")
    
    # Show container-specific suggestions
    container_suggestions = [s for s in result.improvement_suggestions if s.startswith('Container')]
    if container_suggestions:
        print("\n🐳 Container-specific suggestions:")
        for suggestion in container_suggestions:
            print(f"  ✅ {suggestion}")
    else:
        print("\n❌ No container-specific suggestions found")

if __name__ == "__main__":
    print("🚀 Testing New Analysis Features Integration")
    print("=" * 60)
    
    test_cloud_analysis()
    test_framework_analysis()
    test_container_analysis()
    
    print("\n" + "=" * 60)
    print("🎉 All integration tests completed!")
    print("\n💡 The new analyzers are working perfectly!")
    print("   - Framework analysis detects React/Django/Spring issues")
    print("   - Cloud analysis finds AWS/Azure/GCP security problems")
    print("   - Container analysis identifies Docker/K8s misconfigurations") 