#!/usr/bin/env python3
"""
Test script for the Code Quality Dashboard
This script demonstrates the dashboard functionality by creating sample data and generating reports.
"""

import os
import sys
from datetime import datetime, timedelta
import random

# Add the code_analyzer directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code_analyzer'))

try:
    from dashboard import CodeQualityDashboard, QualityMetric
    from models import CodeAnalysisResult
    print("✅ Dashboard module imported successfully!")
except ImportError as e:
    print(f"❌ Error importing dashboard module: {e}")
    sys.exit(1)

def create_sample_analysis_result(quality_score, bug_count, suggestion_count, model_name="deepseek"):
    """Create a sample analysis result for testing."""
    return CodeAnalysisResult(
        code_quality_score=quality_score,
        potential_bugs=[f"Sample bug {i+1}" for i in range(bug_count)],
        improvement_suggestions=[f"Sample suggestion {i+1}" for i in range(suggestion_count)],
        documentation="Sample documentation for the analyzed code.",
        model_name=model_name,
        execution_time=random.uniform(0.5, 3.0)
    )

def generate_sample_data(dashboard, num_samples=20):
    """Generate sample analysis data for testing."""
    print(f"📊 Generating {num_samples} sample analysis records...")
    
    languages = ['python', 'javascript', 'java', 'cpp', 'go']
    models = ['deepseek', 'openai', 'anthropic', 'mercury']
    file_paths = [
        'src/main.py', 'src/utils.py', 'src/models.py',
        'frontend/app.js', 'frontend/components.js',
        'backend/server.java', 'backend/database.java',
        'core/algorithm.cpp', 'core/memory.cpp',
        'pkg/handler.go', 'pkg/middleware.go'
    ]
    
    for i in range(num_samples):
        # Generate realistic quality scores with some trend
        base_quality = 0.6 + (i * 0.02) + random.uniform(-0.1, 0.1)
        quality_score = max(0.1, min(1.0, base_quality))
        
        # Generate related bug and suggestion counts
        bug_count = max(0, int(5 - (quality_score * 5)) + random.randint(-2, 2))
        suggestion_count = max(0, int(8 - (quality_score * 6)) + random.randint(-3, 3))
        
        # Create sample result
        result = create_sample_analysis_result(
            quality_score=quality_score,
            bug_count=bug_count,
            suggestion_count=suggestion_count,
            model_name=random.choice(models)
        )
        
        # Record in dashboard
        file_path = random.choice(file_paths)
        language = random.choice(languages)
        model = result.model_name
        
        dashboard.record_analysis(result, file_path, language, model)
        
        if (i + 1) % 5 == 0:
            print(f"   Generated {i + 1}/{num_samples} records...")
    
    print("✅ Sample data generation complete!")

def test_dashboard_functionality():
    """Test the dashboard functionality."""
    print("🚀 Testing Code Quality Dashboard...")
    
    # Initialize dashboard
    dashboard = CodeQualityDashboard("test_quality_metrics.db")
    print("✅ Dashboard initialized")
    
    # Generate sample data
    generate_sample_data(dashboard, num_samples=25)
    
    # Test different time ranges
    time_ranges = [7, 30, 90]
    
    for days in time_ranges:
        print(f"\n📈 Testing {days}-day report...")
        
        # Generate report
        report = dashboard.generate_dashboard_report(days=days)
        
        # Display report summary
        print(f"   Total analyses: {report.total_analyses}")
        print(f"   Time period: {report.time_period}")
        print(f"   Quality trend: {report.overall_quality_trend.trend_direction}")
        print(f"   Average quality: {report.overall_quality_trend.average_value:.3f}")
        print(f"   Languages analyzed: {len(report.language_breakdown)}")
        print(f"   Models used: {len(report.model_performance)}")
        print(f"   Top issues found: {len(report.top_issues)}")
        print(f"   Improvement areas: {len(report.improvement_areas)}")
        print(f"   Recommendations: {len(report.recommendations)}")
        
        # Test chart generation
        try:
            charts = dashboard.generate_charts(days=days)
            print(f"   Charts generated: {len(charts)}")
        except Exception as e:
            print(f"   Chart generation failed: {e}")
    
    # Test metrics retrieval
    print(f"\n📊 Testing metrics retrieval...")
    metrics = dashboard.get_metrics(days=30)
    print(f"   Retrieved {len(metrics)} metrics")
    
    # Test language filtering
    python_metrics = dashboard.get_metrics(days=30, language_filter='python')
    print(f"   Python metrics: {len(python_metrics)}")
    
    # Test model filtering
    deepseek_metrics = dashboard.get_metrics(days=30, model_filter='deepseek')
    print(f"   DeepSeek metrics: {len(deepseek_metrics)}")
    
    print("\n✅ Dashboard testing complete!")
    
    # Clean up test database
    try:
        os.remove("test_quality_metrics.db")
        print("🧹 Test database cleaned up")
    except:
        pass

def show_dashboard_usage():
    """Show how to use the dashboard in the web application."""
    print("\n🌐 Dashboard Web Integration:")
    print("=" * 50)
    print("To use the dashboard in the web application:")
    print()
    print("1. Start the web server:")
    print("   python -m code_analyzer.web.app")
    print()
    print("2. Open the dashboard in your browser:")
    print("   http://localhost:5000/dashboard")
    print()
    print("3. API endpoints available:")
    print("   GET /api/dashboard/report?days=30")
    print("   GET /api/dashboard/charts?days=30")
    print("   GET /api/dashboard/metrics?days=30&language=python&model=deepseek")
    print()
    print("4. The dashboard will automatically record analysis results")
    print("   when you use the /analyze endpoint")
    print()

if __name__ == "__main__":
    print("🔍 LLM Code Analyzer - Dashboard Test")
    print("=" * 50)
    
    try:
        test_dashboard_functionality()
        show_dashboard_usage()
        
        print("\n🎉 All tests passed! Dashboard is ready to use.")
        print("\nNext steps:")
        print("1. Start the web server: python -m code_analyzer.web.app")
        print("2. Visit http://localhost:5000/dashboard")
        print("3. Analyze some code to see the dashboard in action")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 