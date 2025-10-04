#!/usr/bin/env python3
"""
Health check script for observability stack.
"""

import requests
import sys
import time

def check_service(name, url, timeout=5):
    """Check if a service is healthy."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name}: healthy")
            return True
        else:
            print(f"❌ {name}: unhealthy (status: {response.status_code})")
            return False
    except requests.RequestException as e:
        print(f"❌ {name}: connection failed - {e}")
        return False

def main():
    """Main health check routine."""
    services = [
        ("Grafana", "http://localhost:3000/api/health"),
        ("Prometheus", "http://localhost:9090/-/healthy"),
        ("AlertManager", "http://localhost:9093/-/healthy"),
        ("Loki", "http://localhost:3100/ready"),
        ("Jaeger", "http://localhost:16686/"),
    ]
    
    print("🔭 Checking observability stack health...")
    
    healthy_count = 0
    for name, url in services:
        if check_service(name, url):
            healthy_count += 1
        time.sleep(1)  # Small delay between checks
    
    total_services = len(services)
    print(f"\n📊 Health summary: {healthy_count}/{total_services} services healthy")
    
    if healthy_count == total_services:
        print("🎉 All services are healthy!")
        sys.exit(0)
    elif healthy_count >= total_services * 0.5:
        print("⚠️  Some services are unhealthy but core functionality available")
        sys.exit(0)
    else:
        print("💥 Critical: Most services are unhealthy")
        sys.exit(1)

if __name__ == "__main__":
    main()