#!/usr/bin/env python3
"""
Observability Setup Script for Clinical Data Platform

This script sets up comprehensive observability stack including:
- Prometheus for metrics collection
- Grafana for visualization
- AlertManager for alerting
- Loki for log aggregation
- Jaeger for distributed tracing
"""

import argparse
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import requests


class ObservabilitySetup:
    """Set up and configure observability stack."""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = Path(project_dir)
        self.observability_dir = self.project_dir / "observability"
        self.services = {
            "prometheus": {"port": 9090, "health_path": "/api/v1/status/config"},
            "grafana": {"port": 3000, "health_path": "/api/health"},
            "alertmanager": {"port": 9093, "health_path": "/-/healthy"},
            "loki": {"port": 3100, "health_path": "/ready"},
            "jaeger": {"port": 16686, "health_path": "/"},
        }
        
    def setup_all(self, start_services: bool = True) -> bool:
        """Set up complete observability stack."""
        print("🔭 Setting up Clinical Data Platform Observability Stack")
        print("=" * 60)
        
        setup_tasks = [
            ("📋 Validate configuration files", self.validate_configs),
            ("🔧 Create necessary directories", self.create_directories),
            ("📝 Generate dynamic configs", self.generate_configs),
            ("🐳 Pull Docker images", self.pull_docker_images),
        ]
        
        if start_services:
            setup_tasks.extend([
                ("🚀 Start observability stack", self.start_services),
                ("⏳ Wait for services to be ready", self.wait_for_services),
                ("✅ Validate service health", self.validate_services),
                ("📊 Import Grafana dashboards", self.import_dashboards),
                ("🚨 Configure alerting", self.configure_alerting),
            ])
        
        for task_name, task_func in setup_tasks:
            print(f"\n{task_name}...")
            try:
                success = task_func()
                if success:
                    print(f"✅ {task_name}: SUCCESS")
                else:
                    print(f"❌ {task_name}: FAILED")
                    return False
            except Exception as e:
                print(f"❌ {task_name}: ERROR - {str(e)}")
                return False
        
        self.print_summary()
        return True
    
    def validate_configs(self) -> bool:
        """Validate all configuration files."""
        required_files = [
            "docker-compose.observability.yml",
            "prometheus/prometheus.yml",
            "prometheus/alertmanager.yml",
            "grafana/datasources/prometheus.yml",
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = self.observability_dir / file_path
            if not full_path.exists():
                missing_files.append(str(full_path))
        
        if missing_files:
            print(f"Missing configuration files: {missing_files}")
            return False
        
        # Validate YAML syntax
        yaml_files = [
            "prometheus/prometheus.yml",
            "prometheus/alertmanager.yml", 
            "grafana/datasources/prometheus.yml",
        ]
        
        for yaml_file in yaml_files:
            try:
                with open(self.observability_dir / yaml_file, 'r') as f:
                    yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(f"Invalid YAML in {yaml_file}: {e}")
                return False
        
        return True
    
    def create_directories(self) -> bool:
        """Create necessary directories for observability stack."""
        directories = [
            "prometheus/rules",
            "grafana/dashboards",
            "grafana/config", 
            "loki",
            "data/prometheus",
            "data/grafana",
            "data/loki",
            "data/alertmanager",
        ]
        
        for directory in directories:
            dir_path = self.observability_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created: {dir_path}")
        
        return True
    
    def generate_configs(self) -> bool:
        """Generate dynamic configuration files."""
        
        # Generate Grafana config
        grafana_config = f"""[server]
http_port = 3000
domain = localhost
root_url = http://localhost:3000/

[database]
type = sqlite3
path = grafana.db

[security]
admin_user = admin
admin_password = admin123
secret_key = {self._generate_secret_key()}

[users]
allow_sign_up = false
allow_org_create = false
auto_assign_org = true
auto_assign_org_role = Viewer

[auth.anonymous]
enabled = false

[dashboards]
default_home_dashboard_path = /etc/grafana/provisioning/dashboards/clinical-platform-overview.json

[alerting]
enabled = true

[unified_alerting]
enabled = true

[feature_toggles]
enable = ngalert

[log]
mode = console
level = info
"""
        
        grafana_config_path = self.observability_dir / "grafana/config/grafana.ini"
        with open(grafana_config_path, 'w') as f:
            f.write(grafana_config)
        
        # Generate Loki config
        loki_config = {
            "auth_enabled": False,
            "server": {
                "http_listen_port": 3100,
                "grpc_listen_port": 9096
            },
            "common": {
                "path_prefix": "/loki",
                "storage": {
                    "filesystem": {
                        "chunks_directory": "/loki/chunks",
                        "rules_directory": "/loki/rules"
                    }
                },
                "replication_factor": 1,
                "ring": {
                    "instance_addr": "127.0.0.1",
                    "kvstore": {
                        "store": "inmemory"
                    }
                }
            },
            "query_scheduler": {
                "max_outstanding_requests_per_tenant": 1024
            },
            "schema_config": {
                "configs": [{
                    "from": "2020-10-24",
                    "store": "boltdb-shipper",
                    "object_store": "filesystem",
                    "schema": "v11",
                    "index": {
                        "prefix": "index_",
                        "period": "24h"
                    }
                }]
            },
            "ruler": {
                "alertmanager_url": "http://localhost:9093"
            }
        }
        
        loki_config_path = self.observability_dir / "loki/loki-config.yml"
        with open(loki_config_path, 'w') as f:
            yaml.dump(loki_config, f, default_flow_style=False)
        
        # Generate Promtail config
        promtail_config = {
            "server": {
                "http_listen_port": 9080,
                "grpc_listen_port": 0
            },
            "positions": {
                "filename": "/tmp/positions.yaml"
            },
            "clients": [{
                "url": "http://loki:3100/loki/api/v1/push"
            }],
            "scrape_configs": [{
                "job_name": "containers",
                "static_configs": [{
                    "targets": ["localhost"],
                    "labels": {
                        "job": "containerlogs",
                        "__path__": "/var/lib/docker/containers/*/*log"
                    }
                }],
                "pipeline_stages": [{
                    "json": {
                        "expressions": {
                            "output": "log",
                            "stream": "stream",
                            "attrs": ""
                        }
                    }
                }, {
                    "json": {
                        "expressions": {
                            "tag": "attrs.tag"
                        },
                        "source": "attrs"
                    }
                }, {
                    "regex": {
                        "expression": "(?P<container_name>(?:[^|]*))",
                        "source": "tag"
                    }
                }, {
                    "timestamp": {
                        "format": "RFC3339Nano",
                        "source": "time"
                    }
                }, {
                    "labels": {
                        "stream": "",
                        "container_name": ""
                    }
                }, {
                    "output": {
                        "source": "output"
                    }
                }]
            }]
        }
        
        promtail_config_path = self.observability_dir / "loki/promtail-config.yml"
        with open(promtail_config_path, 'w') as f:
            yaml.dump(promtail_config, f, default_flow_style=False)
        
        return True
    
    def pull_docker_images(self) -> bool:
        """Pull required Docker images."""
        images = [
            "prom/prometheus:v2.47.0",
            "grafana/grafana:10.1.0",
            "prom/alertmanager:v0.26.0",
            "prom/node-exporter:v1.6.1",
            "grafana/loki:2.9.0",
            "grafana/promtail:2.9.0",
            "jaegertracing/all-in-one:1.49",
        ]
        
        for image in images:
            print(f"🐳 Pulling {image}...")
            try:
                result = subprocess.run(
                    ["docker", "pull", image], 
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode != 0:
                    print(f"Failed to pull {image}: {result.stderr}")
                    return False
            except subprocess.TimeoutExpired:
                print(f"Timeout pulling {image}")
                return False
        
        return True
    
    def start_services(self) -> bool:
        """Start observability services using docker-compose."""
        compose_file = self.observability_dir / "docker-compose.observability.yml"
        
        print("🚀 Starting observability stack...")
        try:
            result = subprocess.run([
                "docker-compose", "-f", str(compose_file), "up", "-d"
            ], capture_output=True, text=True, cwd=self.observability_dir)
            
            if result.returncode != 0:
                print(f"Failed to start services: {result.stderr}")
                return False
            
            print("✅ Services started successfully")
            return True
            
        except Exception as e:
            print(f"Error starting services: {e}")
            return False
    
    def wait_for_services(self, timeout: int = 300) -> bool:
        """Wait for all services to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            all_ready = True
            
            for service, config in self.services.items():
                try:
                    url = f"http://localhost:{config['port']}{config['health_path']}"
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code != 200:
                        all_ready = False
                        print(f"⏳ Waiting for {service}...")
                        break
                        
                except requests.RequestException:
                    all_ready = False
                    print(f"⏳ Waiting for {service}...")
                    break
            
            if all_ready:
                print("✅ All services are ready!")
                return True
            
            time.sleep(10)
        
        print(f"❌ Timeout waiting for services after {timeout} seconds")
        return False
    
    def validate_services(self) -> bool:
        """Validate that all services are healthy."""
        failed_services = []
        
        for service, config in self.services.items():
            try:
                url = f"http://localhost:{config['port']}{config['health_path']}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    print(f"✅ {service}: healthy")
                else:
                    print(f"❌ {service}: unhealthy (status: {response.status_code})")
                    failed_services.append(service)
                    
            except requests.RequestException as e:
                print(f"❌ {service}: connection failed - {e}")
                failed_services.append(service)
        
        return len(failed_services) == 0
    
    def import_dashboards(self) -> bool:
        """Import Grafana dashboards."""
        # Wait a bit more for Grafana to be fully ready
        time.sleep(10)
        
        try:
            # Test Grafana API access
            auth = ('admin', 'admin123')
            response = requests.get(
                'http://localhost:3000/api/org',
                auth=auth,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Failed to access Grafana API: {response.status_code}")
                return False
            
            print("✅ Grafana API accessible")
            print("📊 Dashboards will be auto-loaded from provisioning")
            return True
            
        except requests.RequestException as e:
            print(f"Failed to connect to Grafana: {e}")
            return False
    
    def configure_alerting(self) -> bool:
        """Configure alerting rules and notifications."""
        # Check if AlertManager is responding
        try:
            response = requests.get('http://localhost:9093/-/healthy', timeout=10)
            if response.status_code == 200:
                print("✅ AlertManager is healthy and ready")
                return True
            else:
                print(f"❌ AlertManager health check failed: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"Failed to check AlertManager: {e}")
            return False
    
    def print_summary(self):
        """Print setup summary and next steps."""
        print("\n" + "=" * 60)
        print("🎉 OBSERVABILITY STACK SETUP COMPLETE")
        print("=" * 60)
        
        print("\n🌐 Service URLs:")
        service_urls = {
            "Grafana (Dashboards)": "http://localhost:3000",
            "Prometheus (Metrics)": "http://localhost:9090", 
            "AlertManager (Alerts)": "http://localhost:9093",
            "Jaeger (Tracing)": "http://localhost:16686",
        }
        
        for service, url in service_urls.items():
            print(f"  • {service}: {url}")
        
        print("\n🔑 Default Credentials:")
        print("  • Grafana: admin / admin123")
        
        print("\n🚀 Next Steps:")
        next_steps = [
            "Visit Grafana to view Clinical Platform dashboards",
            "Configure notification channels in AlertManager",
            "Start your Clinical Data Platform API to see metrics",
            "Review and customize alerting rules in prometheus/rules/",
            "Set up log forwarding from your applications to Loki"
        ]
        
        for i, step in enumerate(next_steps, 1):
            print(f"  {i}. {step}")
        
        print("\n📚 Documentation:")
        print("  • Prometheus: https://prometheus.io/docs/")
        print("  • Grafana: https://grafana.com/docs/")
        print("  • Loki: https://grafana.com/docs/loki/")
        print("  • Jaeger: https://www.jaegertracing.io/docs/")
    
    def stop_services(self) -> bool:
        """Stop observability services."""
        compose_file = self.observability_dir / "docker-compose.observability.yml"
        
        try:
            result = subprocess.run([
                "docker-compose", "-f", str(compose_file), "down"
            ], capture_output=True, text=True, cwd=self.observability_dir)
            
            if result.returncode == 0:
                print("🛑 Observability stack stopped")
                return True
            else:
                print(f"Failed to stop services: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Error stopping services: {e}")
            return False
    
    def _generate_secret_key(self) -> str:
        """Generate a secret key for Grafana."""
        import secrets
        return secrets.token_hex(32)


def main():
    parser = argparse.ArgumentParser(description="Setup Clinical Data Platform Observability")
    parser.add_argument("--project-dir", default=".", help="Project directory")
    parser.add_argument("--no-start", action="store_true", help="Don't start services")
    parser.add_argument("--stop", action="store_true", help="Stop running services")
    
    args = parser.parse_args()
    
    setup = ObservabilitySetup(args.project_dir)
    
    if args.stop:
        success = setup.stop_services()
    else:
        success = setup.setup_all(start_services=not args.no_start)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()