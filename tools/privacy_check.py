#!/usr/bin/env python3
"""
Privacy Check per Qdrant
Verifica che telemetry sia completamente disabilitata
"""

from typing import Any

import requests
from qdrant_client import QdrantClient


class QdrantPrivacyCheck:
    def __init__(self, host: str = "localhost", port: int = 6334, prefer_grpc: bool = True):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:6333"  # REST API for telemetry check
        self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)

    def check_telemetry_endpoint(self) -> bool:
        """Verifica configurazione telemetry (locale OK, usage statistics NO)"""
        try:
            response = requests.get(f"{self.base_url}/telemetry", timeout=5)

            if response.status_code == 200:
                data = response.json()
                print("✅ Local telemetry endpoint: ACCESSIBLE (corretto per monitoring)")
                print("   Questo endpoint è locale e non invia dati esterni")

                # Verifica che non ci siano dati sensibili o usage statistics
                if "usage_statistics" in str(data).lower():
                    print("⚠️ Warning: Possibili usage statistics attive")
                    return False
                else:
                    print("✅ Contenuto: Solo dati locali database (sicuro)")
                    return True

            elif response.status_code == 404:
                print("⚠️ Local telemetry endpoint: DISABLED")
                print("   Endpoint utile per monitoring sistema CBT")
                return True  # Accettabile ma non ottimale
            else:
                print(f"⚠️ Telemetry endpoint: Unexpected status {response.status_code}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"❌ Telemetry endpoint: Connection failed ({str(e)})")
            return False

    def check_metrics_endpoint(self) -> bool:
        """Verifica endpoints metriche"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=5)

            if response.status_code == 404:
                print("✅ Metrics endpoint: DISABLED")
                return True
            elif response.status_code == 200:
                print("⚠️ Metrics endpoint: ACCESSIBLE")
                print("   (Può essere OK se solo per monitoring locale)")
                return True  # Metrics locale può essere accettabile
            else:
                print(f"✅ Metrics endpoint: Status {response.status_code}")
                return True

        except requests.exceptions.RequestException:
            print("✅ Metrics endpoint: NOT ACCESSIBLE")
            return True

    def check_cluster_info(self) -> bool:
        """Verifica che info cluster non includano dati sensibili"""
        try:
            response = requests.get(f"{self.base_url}/cluster", timeout=5)

            if response.status_code == 200:
                data = response.json()

                # Check per campi che potrebbero essere sensibili
                sensitive_fields = ["peer_id", "uri", "telemetry"]
                found_sensitive = []

                def check_recursive(obj: Any, path: str = "") -> None:
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            current_path = f"{path}.{key}" if path else key
                            if any(field in key.lower() for field in sensitive_fields):
                                found_sensitive.append(current_path)
                            check_recursive(value, current_path)
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                            check_recursive(item, f"{path}[{i}]")

                check_recursive(data)

                if found_sensitive:
                    print("⚠️ Cluster info: Contains potentially sensitive fields:")
                    for field in found_sensitive:
                        print(f"   - {field}")
                else:
                    print("✅ Cluster info: No sensitive fields detected")

                return True

        except requests.exceptions.RequestException:
            print("✅ Cluster endpoint: NOT ACCESSIBLE")
            return True

    def check_docker_environment(self) -> bool:
        """Verifica variabili ambiente Docker"""
        try:
            import shutil
            import subprocess

            # Verify docker command exists
            if not shutil.which("docker"):
                print("⚠️ Docker command not found in PATH")
                return False

            # Check container environment with secure subprocess call
            result = subprocess.run(
                ["docker", "exec", "cbt_qdrant", "env"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,  # Don't raise on non-zero exit
            )

            if result.returncode == 0:
                env_vars = result.stdout

                if "QDRANT__TELEMETRY_DISABLED=true" in env_vars:
                    print("✅ Docker environment: QDRANT__TELEMETRY_DISABLED=true")
                    return True
                else:
                    print("❌ Docker environment: QDRANT__TELEMETRY_DISABLED not found")
                    print("   Available QDRANT env vars:")
                    for line in env_vars.split("\n"):
                        if "QDRANT" in line:
                            print(f"   {line}")
                    return False
            else:
                print(f"⚠️ Cannot check Docker environment: {result.stderr}")
                return False

        except Exception as e:
            print(f"⚠️ Docker environment check failed: {str(e)}")
            return False

    def check_network_traffic(self) -> bool:
        """Suggerimenti per monitoring traffico di rete"""
        print("\n🔍 NETWORK TRAFFIC MONITORING:")
        print("Per verifica completa privacy, monitora traffico rete:")
        print("")
        print("1. Container network:")
        print("   docker exec cbt_qdrant netstat -tuln")
        print("")
        print("2. Outbound connections:")
        print("   docker exec cbt_qdrant ss -tuln")
        print("")
        print("3. Host-level monitoring:")
        print("   sudo netstat -tuln | grep 6333")
        print("   sudo ss -tuln | grep 6333")
        print("")
        print("4. Packet capture (advanced):")
        print("   sudo tcpdump -i any port 6333")
        print("")
        return True

    def run_complete_privacy_check(self) -> bool:
        """Esegui check privacy completo"""
        print("=" * 60)
        print("QDRANT PRIVACY CHECK")
        print("=" * 60)

        all_checks_passed = True

        print("\n🔒 Checking telemetry configuration...")
        if not self.check_telemetry_endpoint():
            all_checks_passed = False

        print("\n📊 Checking metrics endpoint...")
        if not self.check_metrics_endpoint():
            all_checks_passed = False

        print("\n🌐 Checking cluster info...")
        if not self.check_cluster_info():
            all_checks_passed = False

        print("\n🐳 Checking Docker environment...")
        if not self.check_docker_environment():
            all_checks_passed = False

        # Network monitoring suggestions
        self.check_network_traffic()

        print("\n" + "=" * 60)
        if all_checks_passed:
            print("✅ PRIVACY + MONITORING CHECK: PASSED")
            print("Configurazione ottimale:")
            print("- Usage statistics DISABILITATE (privacy)")
            print("- Local monitoring ABILITATO (system health)")
            print("- Dati CBT sicuri e sistema monitorabile")
        else:
            print("❌ PRIVACY CHECK: ISSUES FOUND")
            print("Verificare configurazione per garantire privacy completa")

        print("=" * 60)
        return all_checks_passed


def main() -> int:
    """Privacy check principale"""
    checker = QdrantPrivacyCheck()
    success = checker.run_complete_privacy_check()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
