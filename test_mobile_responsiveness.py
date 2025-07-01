#!/usr/bin/env python3
"""
Test script to verify mobile responsiveness features
"""

import requests
import time
import json

def test_mobile_responsiveness():
    """Test mobile responsiveness features"""
    base_url = "http://localhost:5000"
    
    print("🤖 Testing Mobile Responsiveness Features")
    print("=" * 50)
    
    # Test 1: Check if the main page loads with mobile viewport
    print("\n1. Testing main page mobile viewport...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            if 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no' in response.text:
                print("✅ Mobile viewport meta tag found")
            else:
                print("❌ Mobile viewport meta tag missing")
        else:
            print(f"❌ Failed to load main page: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing main page: {e}")
    
    # Test 2: Check if mobile-specific CSS classes are present
    print("\n2. Testing mobile-specific CSS classes...")
    try:
        response = requests.get(f"{base_url}/")
        mobile_classes = [
            'mobile-nav-toggle',
            'mobile-fab',
            'touch-active',
            'loading-spinner',
            '@media (max-width: 768px)',
            '@media (max-width: 767px)'
        ]
        
        found_classes = []
        for class_name in mobile_classes:
            if class_name in response.text:
                found_classes.append(class_name)
        
        if len(found_classes) >= 4:
            print(f"✅ Found {len(found_classes)} mobile-specific classes: {', '.join(found_classes[:4])}")
        else:
            print(f"❌ Only found {len(found_classes)} mobile classes: {found_classes}")
    except Exception as e:
        print(f"❌ Error testing mobile CSS: {e}")
    
    # Test 3: Test dashboard mobile responsiveness
    print("\n3. Testing dashboard mobile responsiveness...")
    try:
        response = requests.get(f"{base_url}/dashboard")
        if response.status_code == 200:
            if 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no' in response.text:
                print("✅ Dashboard has mobile viewport")
            else:
                print("❌ Dashboard missing mobile viewport")
                
            if 'stats-grid' in response.text and 'grid-template-columns' in response.text:
                print("✅ Dashboard has responsive grid layout")
            else:
                print("❌ Dashboard missing responsive grid")
        else:
            print(f"❌ Failed to load dashboard: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
    
    # Test 4: Test mobile-specific JavaScript features
    print("\n4. Testing mobile JavaScript features...")
    try:
        response = requests.get(f"{base_url}/")
        mobile_js_features = [
            'mobile-nav-toggle',
            'mobile-fab',
            'hapticFeedback',
            'orientationchange',
            'touchstart',
            'touchend',
            'preventDefault'
        ]
        
        found_features = []
        for feature in mobile_js_features:
            if feature in response.text:
                found_features.append(feature)
        
        if len(found_features) >= 5:
            print(f"✅ Found {len(found_features)} mobile JS features: {', '.join(found_features[:5])}")
        else:
            print(f"❌ Only found {len(found_features)} mobile JS features: {found_features}")
    except Exception as e:
        print(f"❌ Error testing mobile JavaScript: {e}")
    
    # Test 5: Test touch-friendly button sizes
    print("\n5. Testing touch-friendly button sizes...")
    try:
        response = requests.get(f"{base_url}/")
        if 'min-height: 44px' in response.text:
            print("✅ Touch-friendly button sizes found")
        else:
            print("❌ Touch-friendly button sizes missing")
    except Exception as e:
        print(f"❌ Error testing button sizes: {e}")
    
    # Test 6: Test responsive breakpoints
    print("\n6. Testing responsive breakpoints...")
    try:
        response = requests.get(f"{base_url}/")
        breakpoints = [
            '@media (max-width: 768px)',
            '@media (min-width: 768px)',
            '@media (min-width: 1024px)'
        ]
        
        found_breakpoints = []
        for bp in breakpoints:
            if bp in response.text:
                found_breakpoints.append(bp)
        
        if len(found_breakpoints) >= 2:
            print(f"✅ Found {len(found_breakpoints)} responsive breakpoints: {', '.join(found_breakpoints)}")
        else:
            print(f"❌ Only found {len(found_breakpoints)} breakpoints: {found_breakpoints}")
    except Exception as e:
        print(f"❌ Error testing breakpoints: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Mobile Responsiveness Test Complete!")
    print("\n📱 Key Mobile Features Added:")
    print("• Mobile-first responsive design")
    print("• Touch-friendly button sizes (44px minimum)")
    print("• Mobile navigation hamburger menu")
    print("• Floating action button for quick actions")
    print("• Haptic feedback support")
    print("• Auto-resizing textareas")
    print("• Orientation change handling")
    print("• Mobile-optimized copy functionality")
    print("• Loading states and animations")
    print("• Accessibility improvements")
    print("• High DPI display support")
    print("• Reduced motion support")

if __name__ == "__main__":
    test_mobile_responsiveness() 