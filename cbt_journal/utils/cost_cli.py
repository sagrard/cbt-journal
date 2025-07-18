#!/usr/bin/env python3
"""
CLI Tool per Cost Control Management
Permette testing, monitoring e gestione budget da command line
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Aggiungi path per import locale
sys.path.append(str(Path(__file__).parent.parent))

from cbt_journal.utils.cost_control import CostControlManager


class CostControlCLI:
    """Command Line Interface per Cost Control"""

    def __init__(self) -> None:
        self.cost_manager = CostControlManager()

    def check_budget(
        self,
        session_id: str,
        tokens_input: int,
        tokens_output: int,
        model: str = "gpt-4o-2024-11-20",
    ) -> None:
        """Comando: check budget pre-API call"""

        print(f"🔍 Checking budget for session: {session_id}")
        print(f"📊 Estimated usage: {tokens_input} input + {tokens_output} output tokens")
        print(f"🤖 Model: {model}")
        print("-" * 60)

        allowed, result = self.cost_manager.pre_api_check(
            session_id=session_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model,
        )

        # Session budget status
        session = result["session_budget"]
        print("💰 SESSION BUDGET:")
        print(f"   Current cost: ${session['current_session_cost']:.6f}")
        print(f"   Limit: ${session['max_cost_per_session']:.2f}")
        print(f"   Utilization: {session['budget_utilization']:.1%}")
        print(f"   Status: {session['budget_status']}")

        # Daily budget status
        daily = result["daily_tracking"]
        print("\n📅 DAILY BUDGET:")
        print(f"   Current cost: ${daily['current_daily_cost']:.4f}")
        print(f"   Limit: ${daily['daily_cost_limit']:.2f}")
        print(f"   Sessions today: {daily['sessions_today']}")
        print(f"   Avg per session: ${daily['avg_cost_per_session']:.4f}")
        print(f"   Status: {daily['daily_budget_status']}")

        # Monthly budget status
        monthly = result["monthly_tracking"]
        print("\n📆 MONTHLY BUDGET:")
        print(f"   Current cost: ${monthly['current_monthly_cost']:.4f}")
        print(f"   Limit: ${monthly['monthly_budget']:.2f}")
        print(f"   Utilization: {monthly['budget_utilization']:.1%}")
        print(f"   Days elapsed: {monthly['days_elapsed']}")
        print(f"   Status: {monthly['monthly_budget_status']}")

        # Alerts
        alerts = result["cost_alerts"]
        if any(
            [
                alerts["session_over_budget"],
                alerts["daily_approaching_limit"],
                alerts["monthly_approaching_limit"],
                alerts["unusual_cost_spike"],
            ]
        ):
            print("\n⚠️  ALERTS:")
            if alerts["session_over_budget"]:
                print("   🚨 Session over budget!")
            if alerts["daily_approaching_limit"]:
                print("   ⚠️  Daily budget approaching limit")
            if alerts["monthly_approaching_limit"]:
                print("   ⚠️  Monthly budget approaching limit")
            if alerts["unusual_cost_spike"]:
                print("   📈 Unusual cost spike detected")

        # Optimization suggestions
        suggestions = result["optimization_suggestions"]
        if suggestions:
            print("\n💡 OPTIMIZATION SUGGESTIONS:")
            for i, suggestion in enumerate(suggestions, 1):
                print(f"   {i}. {suggestion['description']}")
                print(f"      Potential savings: ${suggestion['potential_savings_usd']:.4f}")
                print(f"      Confidence: {suggestion['confidence']}")

        # Final verdict
        print("\n" + "=" * 60)
        if allowed:
            print("✅ API CALL ALLOWED")
            print(f"💸 Estimated cost: ${result['estimated_cost']:.6f}")
        else:
            print("❌ API CALL BLOCKED")
            print("🚫 Budget limits exceeded")

        return allowed

    def record_cost(
        self,
        session_id: str,
        api_type: str,
        model: str,
        tokens_input: int,
        tokens_output: int,
        actual_cost: float,
        purpose: Optional[str] = None,
    ) -> None:
        """Comando: registra costo API effettivo"""

        self.cost_manager.record_api_cost(
            session_id=session_id,
            api_type=api_type,
            model=model,
            provider="openai",  # Default per ora
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            actual_cost=actual_cost,
            purpose=purpose,
        )

        print(f"✅ Cost recorded: ${actual_cost:.6f} for session {session_id}")

        # Show updated daily summary
        summary = self.cost_manager.get_cost_summary("today")
        print(f"📊 Today's total: ${summary['total_cost']:.4f} / ${summary['budget_limit']:.2f}")
        print(f"📈 Utilization: {summary['utilization']:.1%}")

    def show_summary(self, period: str = "today") -> None:
        """Comando: mostra summary costi"""

        summary = self.cost_manager.get_cost_summary(period)

        print(f"📊 COST SUMMARY ({period.upper()})")
        print("=" * 50)

        if period == "today":
            print(f"💰 Total cost: ${summary['total_cost']:.4f}")
            print(f"🎯 Budget limit: ${summary['budget_limit']:.2f}")
            print(f"📈 Utilization: {summary['utilization']:.1%}")
            print(f"📱 Sessions: {summary['sessions']}")

            if summary["sessions"] > 0:
                avg_per_session = summary["total_cost"] / summary["sessions"]
                print(f"💵 Avg per session: ${avg_per_session:.4f}")

            # Status indicator
            if summary["utilization"] >= 1.0:
                print("🚨 STATUS: OVER BUDGET")
            elif summary["utilization"] >= 0.8:
                print("⚠️  STATUS: APPROACHING LIMIT")
            else:
                print("✅ STATUS: WITHIN LIMITS")

        elif period == "month":
            print(f"💰 Total cost: ${summary['total_cost']:.4f}")
            print(f"🎯 Budget limit: ${summary['budget_limit']:.2f}")
            print(f"📈 Utilization: {summary['utilization']:.1%}")
            print(f"📅 Days elapsed: {summary['days_elapsed']}")

            if summary["days_elapsed"] > 0:
                daily_avg = summary["total_cost"] / summary["days_elapsed"]
                days_remaining = 30 - summary["days_elapsed"]
                projected = summary["total_cost"] + (daily_avg * days_remaining)

                print(f"💵 Daily average: ${daily_avg:.4f}")
                print(f"🔮 Projected month: ${projected:.2f}")

                if projected > summary["budget_limit"]:
                    print("⚠️  PROJECTION: WILL EXCEED BUDGET")
                else:
                    print("✅ PROJECTION: WITHIN BUDGET")

    def estimate_cost(
        self, tokens_input: int, tokens_output: int, model: str = "gpt-4o-2024-11-20"
    ) -> None:
        """Comando: stima costo senza registrare"""

        cost = self.cost_manager.estimate_api_cost(tokens_input, tokens_output, model)

        print("💰 COST ESTIMATION")
        print("-" * 30)
        print(f"🤖 Model: {model}")
        print(f"📥 Input tokens: {tokens_input:,}")
        print(f"📤 Output tokens: {tokens_output:,}")
        print(f"💸 Estimated cost: ${cost:.6f}")

        # Show as percentage of budgets
        session_pct = (cost / self.cost_manager.max_cost_per_session) * 100
        daily_pct = (cost / self.cost_manager.max_daily_cost) * 100
        monthly_pct = (cost / self.cost_manager.max_monthly_cost) * 100

        print("\n📊 BUDGET IMPACT:")
        print(f"   Session budget: {session_pct:.1f}%")
        print(f"   Daily budget: {daily_pct:.1f}%")
        print(f"   Monthly budget: {monthly_pct:.2f}%")

    def set_limits(
        self, session_limit: Optional[float] = None, daily_limit: Optional[float] = None, monthly_limit: Optional[float] = None
    ) -> None:
        """Comando: modifica limiti budget"""

        print("🎛️  UPDATING BUDGET LIMITS")
        print("-" * 30)

        if session_limit is not None:
            old_limit = self.cost_manager.max_cost_per_session
            self.cost_manager.max_cost_per_session = session_limit
            print(f"💰 Session limit: ${old_limit:.2f} → ${session_limit:.2f}")

        if daily_limit is not None:
            old_limit = self.cost_manager.max_daily_cost
            self.cost_manager.max_daily_cost = daily_limit
            print(f"📅 Daily limit: ${old_limit:.2f} → ${daily_limit:.2f}")

        if monthly_limit is not None:
            old_limit = self.cost_manager.max_monthly_cost
            self.cost_manager.max_monthly_cost = monthly_limit
            print(f"📆 Monthly limit: ${old_limit:.2f} → ${monthly_limit:.2f}")

        print("\n✅ Limits updated successfully")
        print("💡 Note: Changes apply to new budget checks only")

    def show_alerts(self) -> None:
        """Comando: mostra alert recenti"""

        import sqlite3

        with sqlite3.connect(self.cost_manager.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT timestamp, alert_type, severity, message
                FROM cost_alerts
                ORDER BY timestamp DESC
                LIMIT 10
            """
            )
            alerts = cursor.fetchall()

        if not alerts:
            print("✅ No recent alerts")
            return

        print("⚠️  RECENT ALERTS")
        print("=" * 50)

        for timestamp, alert_type, severity, message in alerts:
            # Parse timestamp for better display
            dt = datetime.fromisoformat(timestamp)
            time_str = dt.strftime("%Y-%m-%d %H:%M")

            # Emoji per severity
            emoji = {"ERROR": "🚨", "CRITICAL": "💥", "WARNING": "⚠️", "INFO": "ℹ️"}.get(
                severity, "📝"
            )

            print(f"{emoji} [{time_str}] {severity}")
            print(f"   Type: {alert_type}")
            print(f"   Message: {message}")
            print()

    def test_scenario(self, scenario: str) -> None:
        """Comando: test scenari predefiniti"""

        scenarios = {
            "light": {
                "description": "Light usage scenario",
                "tokens_input": 500,
                "tokens_output": 200,
                "model": "gpt-4o-2024-11-20",
            },
            "normal": {
                "description": "Normal CBT session",
                "tokens_input": 2000,
                "tokens_output": 800,
                "model": "gpt-4o-2024-11-20",
            },
            "heavy": {
                "description": "Heavy session with context",
                "tokens_input": 8000,
                "tokens_output": 2000,
                "model": "gpt-4o-2024-11-20",
            },
            "mini": {
                "description": "Using mini model",
                "tokens_input": 2000,
                "tokens_output": 800,
                "model": "gpt-4o-mini",
            },
        }

        if scenario not in scenarios:
            print(f"❌ Unknown scenario: {scenario}")
            print(f"Available scenarios: {', '.join(scenarios.keys())}")
            return

        config = scenarios[scenario]
        print(f"🧪 Testing scenario: {config['description']}")
        print("-" * 50)

        self.check_budget(
            session_id=f"test_scenario_{scenario}",
            tokens_input=config["tokens_input"],
            tokens_output=config["tokens_output"],
            model=config["model"],
        )

    def export_data(self, output_file: str) -> None:
        """Comando: export dati per analisi"""

        import sqlite3
        import csv

        print(f"📊 Exporting cost data to: {output_file}")

        with sqlite3.connect(self.cost_manager.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT session_id, timestamp, api_type, model, provider,
                       tokens_input, tokens_output, cost_usd, purpose
                FROM api_costs
                ORDER BY timestamp DESC
            """
            )

            with open(output_file, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                # Header
                writer.writerow(
                    [
                        "session_id",
                        "timestamp",
                        "api_type",
                        "model",
                        "provider",
                        "tokens_input",
                        "tokens_output",
                        "cost_usd",
                        "purpose",
                    ]
                )

                # Data
                count = 0
                for row in cursor:
                    writer.writerow(row)
                    count += 1

        print(f"✅ Exported {count} records")
        print(f"📁 File: {output_file}")


def main() -> int:
    """Main CLI function"""

    parser = argparse.ArgumentParser(
        description="CBT Journal Cost Control CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check budget before API call
  python cost_control_cli.py check --session test_001 --input 1000 --output 500

  # Record actual cost
  python cost_control_cli.py record --session test_001 --type chat --model gpt-4o --input 1000 --output 500 --cost 0.0075

  # Show daily summary
  python cost_control_cli.py summary --period today

  # Estimate cost for planning
  python cost_control_cli.py estimate --input 2000 --output 800

  # Test predefined scenarios
  python cost_control_cli.py test --scenario normal

  # Show recent alerts
  python cost_control_cli.py alerts
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check budget before API call")
    check_parser.add_argument("--session", required=True, help="Session ID")
    check_parser.add_argument("--input", type=int, required=True, help="Input tokens")
    check_parser.add_argument("--output", type=int, required=True, help="Output tokens")
    check_parser.add_argument("--model", default="gpt-4o-2024-11-20", help="Model name")

    # Record command
    record_parser = subparsers.add_parser("record", help="Record actual API cost")
    record_parser.add_argument("--session", required=True, help="Session ID")
    record_parser.add_argument("--type", required=True, help="API type (e.g., chat, embedding)")
    record_parser.add_argument("--model", required=True, help="Model name")
    record_parser.add_argument("--input", type=int, required=True, help="Input tokens")
    record_parser.add_argument("--output", type=int, required=True, help="Output tokens")
    record_parser.add_argument("--cost", type=float, required=True, help="Actual cost in USD")
    record_parser.add_argument("--purpose", help="API call purpose")

    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Show cost summary")
    summary_parser.add_argument(
        "--period", choices=["today", "month"], default="today", help="Period"
    )

    # Estimate command
    estimate_parser = subparsers.add_parser("estimate", help="Estimate API cost")
    estimate_parser.add_argument("--input", type=int, required=True, help="Input tokens")
    estimate_parser.add_argument("--output", type=int, required=True, help="Output tokens")
    estimate_parser.add_argument("--model", default="gpt-4o-2024-11-20", help="Model name")

    # Set limits command
    limits_parser = subparsers.add_parser("limits", help="Set budget limits")
    limits_parser.add_argument("--session", type=float, help="Session limit USD")
    limits_parser.add_argument("--daily", type=float, help="Daily limit USD")
    limits_parser.add_argument("--monthly", type=float, help="Monthly limit USD")

    # Alerts command
    subparsers.add_parser("alerts", help="Show recent alerts")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test predefined scenarios")
    test_parser.add_argument(
        "--scenario",
        choices=["light", "normal", "heavy", "mini"],
        required=True,
        help="Test scenario",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export cost data")
    export_parser.add_argument("--output", default="cost_data.csv", help="Output file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Initialize CLI
    cli = CostControlCLI()

    try:
        if args.command == "check":
            allowed = cli.check_budget(args.session, args.input, args.output, args.model)
            return 0 if allowed else 1

        elif args.command == "record":
            cli.record_cost(
                args.session,
                args.type,
                args.model,
                args.input,
                args.output,
                args.cost,
                args.purpose,
            )

        elif args.command == "summary":
            cli.show_summary(args.period)

        elif args.command == "estimate":
            cli.estimate_cost(args.input, args.output, args.model)

        elif args.command == "limits":
            cli.set_limits(args.session, args.daily, args.monthly)

        elif args.command == "alerts":
            cli.show_alerts()

        elif args.command == "test":
            cli.test_scenario(args.scenario)

        elif args.command == "export":
            cli.export_data(args.output)

        return 0

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
