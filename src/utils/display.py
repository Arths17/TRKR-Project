"""
Display Module - Formats and displays predictions in the terminal
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from typing import Dict, List
import pandas as pd
from datetime import datetime

import config


class F1Display:
    """Handles terminal display of predictions and insights"""
    
    def __init__(self):
        """Initialize the display handler"""
        self.console = Console()
    
    def show_banner(self):
        """Display application banner"""
        banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║        🏎️  F1 RACE PREDICTION SYSTEM  🏁                     ║
║                                                               ║
║           AI-Powered Formula 1 Analytics                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
        """
        self.console.print(banner, style="bold cyan")
    
    def show_race_header(self, race_name: str, circuit: str, date: str = None):
        """
        Display race header information
        
        Args:
            race_name: Name of the race
            circuit: Circuit name
            date: Race date (optional)
        """
        header_text = f"\n[bold yellow]Race:[/bold yellow] {race_name}\n"
        header_text += f"[bold yellow]Circuit:[/bold yellow] {circuit}\n"
        if date:
            header_text += f"[bold yellow]Date:[/bold yellow] {date}\n"
        
        panel = Panel(
            header_text,
            title="Race Information",
            border_style="yellow",
            box=box.ROUNDED
        )
        self.console.print(panel)
    
    def show_predictions_table(self, predictions: pd.DataFrame, top_n: int = None):
        """
        Display predictions in a formatted table
        
        Args:
            predictions: DataFrame with predictions
            top_n: Number of top predictions to show
        """
        top_n = top_n or config.TOP_N_PREDICTIONS
        
        # Create table
        table = Table(
            title="Race Predictions (Ranked by Predicted Race Time)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        
        # Add columns
        table.add_column("Pos", style="cyan", width=4)
        table.add_column("Driver", style="bold white", width=20)
        table.add_column("Team", style="green", width=25)
        table.add_column("Pred Time (s)", justify="right", style="yellow", width=14)
        table.add_column("Gap", justify="right", style="bright_black", width=10)
        
        # Get winner time for gap calculation
        winner_time = predictions.iloc[0]['PredictedRaceTime'] if len(predictions) > 0 else 0
        
        # Add rows
        for idx, (_, row) in enumerate(predictions.head(top_n).iterrows(), 1):
            gap = row.get('Gap', row['PredictedRaceTime'] - winner_time)
            gap_str = f"+{gap:.2f}s" if gap > 0 else "-"
            
            row_data = [
                str(idx), 
                row['Driver'], 
                row['Team'],
                f"{row.get('PredictedRaceTime', float('nan')):.2f}",
                gap_str
            ]
            
            # Highlight top 3
            style = "bold" if idx <= 3 else None
            table.add_row(*row_data, style=style)
        
        self.console.print("\n")
        self.console.print(table)
    
    def show_winner_prediction(self, insights: Dict):
        """
        Display predicted winner in a highlighted panel
        
        Args:
            insights: Dictionary with insights including predicted winner
        """
        winner = insights.get('predicted_winner', {})
        
        if not winner:
            return
        
        winner_text = f"""
    [bold yellow]Driver:[/bold yellow] {winner['driver']}
    [bold yellow]Team:[/bold yellow] {winner['team']}
    [bold yellow]Predicted Race Time:[/bold yellow] {winner['predicted_time']:.2f}s
        """
        
        panel = Panel(
            winner_text,
            title="🏆 Predicted Winner",
            border_style="bold gold1",
            box=box.DOUBLE
        )
        
        self.console.print("\n")
        self.console.print(panel)
    
    def show_podium_prediction(self, insights: Dict):
        """
        Display predicted podium finishers
        
        Args:
            insights: Dictionary with insights including predicted podium
        """
        podium = insights.get('predicted_podium', [])
        
        if not podium or len(podium) < 3:
            return
        
        # Create podium table
        table = Table(
            title="🥇 🥈 🥉 Predicted Podium",
            box=box.HEAVY,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Position", style="yellow", width=10)
        table.add_column("Driver", style="bold white", width=20)
        table.add_column("Team", style="green", width=25)
        table.add_column("Pred Time (s)", justify="right", style="cyan", width=15)
        table.add_column("Gap", justify="right", style="bright_black", width=10)
        
        positions = ["🥇 P1", "🥈 P2", "🥉 P3"]
        
        winner_time = podium[0]['predicted_time'] if podium else 0
        for pos, driver in zip(positions[:len(podium)], podium):
            gap = driver['predicted_time'] - winner_time
            gap_str = f"+{gap:.2f}s" if gap > 0 else "-"
            table.add_row(
                pos,
                driver['driver'],
                driver['team'],
                f"{driver['predicted_time']:.2f}",
                gap_str
            )
        
        self.console.print("\n")
        self.console.print(table)
    
    def show_insights(self, insights: Dict):
        """
        Display additional insights and analysis
        
        Args:
            insights: Dictionary with various insights
        """
        # Dark horses
        dark_horses = insights.get('dark_horses', [])
        if dark_horses:
            self.console.print("\n[bold cyan]Dark Horses (Close to Podium):[/bold cyan]")
            for dh in dark_horses[:3]:
                self.console.print(
                    f"  • {dh['driver']} ({dh['team']}) - "
                    f"Predicted: {dh['predicted_time']:.2f}s, Gap: {dh['gap_to_podium']:.2f}s"
                )
        
        # Strongest teams
        strongest_teams = insights.get('strongest_teams', {})
        if strongest_teams:
            self.console.print("\n[bold cyan]Strongest Teams (Team Strength Index 0-100):[/bold cyan]")
            for idx, (team, strength) in enumerate(list(strongest_teams.items())[:3], 1):
                # Display strength as bar graph
                bar_length = int(strength / 5)  # 100 -> 20 chars
                bar = "█" * bar_length
                self.console.print(f"  {idx}. {team:25s} [{strength:5.1f}] {bar}")
        
        # Fastest average laps
        fastest_laps = insights.get('fastest_avg_lap', [])
        if fastest_laps:
            self.console.print("\n[bold cyan]Fastest Average Lap Times:[/bold cyan]")
            for fl in fastest_laps:
                self.console.print(
                    f"  • {fl['driver']} ({fl['team']}) - {fl['avg_lap_time']:.2f}s"
                )
        
        # Validation summary
        validation = insights.get('validation', {})
        confidence = insights.get('confidence', {})
        if validation or confidence:
            self.show_validation_summary(validation, confidence)
    
    def show_data_summary(self, data_summary: Dict):
        """
        Display summary of data used for predictions
        
        Args:
            data_summary: Dictionary with data statistics
        """
        summary_text = f"""
[bold yellow]Races Analyzed:[/bold yellow] {data_summary.get('total_races', 0)}
[bold yellow]Seasons:[/bold yellow] {data_summary.get('seasons', 'N/A')}
[bold yellow]Total Laps:[/bold yellow] {data_summary.get('total_laps', 0):,}
[bold yellow]Drivers:[/bold yellow] {data_summary.get('unique_drivers', 0)}
[bold yellow]Model Type:[/bold yellow] {data_summary.get('model_type', config.MODEL_TYPE).upper()}
        """
        
        panel = Panel(
            summary_text,
            title="Data Summary",
            border_style="blue",
            box=box.ROUNDED
        )
        
        self.console.print("\n")
        self.console.print(panel)
    
    def show_model_performance(self, performance: Dict):
        """
        Display model performance metrics
        
        Args:
            performance: Dictionary with model performance data
        """
        if not performance:
            return
        
        perf_text = ""
        for target, metrics in performance.items():
            accuracy = metrics.get('accuracy', 0)
            perf_text += f"[bold yellow]{target}:[/bold yellow] {accuracy*100:.1f}% accuracy\n"
        
        panel = Panel(
            perf_text,
            title="Model Performance",
            border_style="green",
            box=box.ROUNDED
        )
        
        self.console.print("\n")
        self.console.print(panel)
    
    def show_error(self, message: str):
        """
        Display error message
        
        Args:
            message: Error message to display
        """
        self.console.print(f"\n[bold red]Error:[/bold red] {message}\n")
    
    def show_warning(self, message: str):
        """
        Display warning message
        
        Args:
            message: Warning message to display
        """
        self.console.print(f"\n[bold yellow]Warning:[/bold yellow] {message}\n")
    
    def show_info(self, message: str):
        """
        Display info message
        
        Args:
            message: Info message to display
        """
        self.console.print(f"\n[cyan]ℹ[/cyan] {message}\n")
    
    def show_progress(self, message: str):
        """
        Display progress message
        
        Args:
            message: Progress message to display
        """
        self.console.print(f"[bold green]→[/bold green] {message}")
    
    def show_validation_summary(self, validation: Dict, confidence: Dict = None):
        """
        Display validation summary for prediction integrity with confidence scoring
        
        Args:
            validation: Dictionary with validation results
            confidence: Dictionary with confidence scoring (optional)
        """
        self.console.print("\n")
        
        # Create validation table
        table = Table(
            title="🔍 Validation & Confidence Summary",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Check", style="white", width=30)
        table.add_column("Status", style="bold", width=20)
        
        # Sorted correctness
        sorted_status = "✅ PASS" if validation.get('sorted_correct', False) else "❌ FAIL"
        sorted_style = "green" if validation.get('sorted_correct', False) else "red"
        table.add_row("Ranking Integrity", sorted_status, style=sorted_style)
        
        # Gap validity
        gaps_status = "✅ PASS" if validation.get('gaps_valid', False) else "❌ FAIL"
        gaps_style = "green" if validation.get('gaps_valid', False) else "red"
        table.add_row("Gap Validity", gaps_status, style=gaps_style)
        
        # Feature coverage
        if 'feature_coverage' in validation:
            coverage = validation['feature_coverage']
            coverage_str = f"{coverage:.1f}%"
            coverage_style = "green" if coverage >= 85 else ("yellow" if coverage >= 70 else "red")
            table.add_row("Feature Coverage", coverage_str, style=coverage_style)
        
        # Confidence score (NEW)
        if confidence:
            conf_level = confidence.get('level', 'UNKNOWN')
            conf_score = confidence.get('score', 0)
            conf_str = f"{conf_level} ({conf_score:.0f}/100)"
            
            if conf_level == 'HIGH':
                conf_style = "bold green"
                conf_str = f"⭐ {conf_str}"
            elif conf_level == 'MEDIUM':
                conf_style = "bold yellow"
                conf_str = f"⚠️ {conf_str}"
            else:
                conf_style = "bold red"
                conf_str = f"❌ {conf_str}"
            
            table.add_row("Prediction Confidence", conf_str, style=conf_style)
        
        # Overall status
        overall_status = validation.get('status', 'UNKNOWN')
        overall_style = "bold green" if overall_status == 'PASS' else "bold red"
        table.add_row("Overall", f"{'✅' if overall_status == 'PASS' else '❌'} {overall_status}", style=overall_style)
        
        self.console.print(table)
        
        # Show errors if any
        errors = validation.get('errors', [])
        if errors:
            self.console.print("\n[bold red]Validation Errors:[/bold red]")
            for error in errors:
                self.console.print(f"  ⚠️  {error}")
        
        # Show confidence explanation if LOW
        if confidence and confidence.get('level') == 'LOW':
            self.console.print("\n[bold yellow]Low Confidence Explanation:[/bold yellow]")
            self.console.print(f"  - Feature coverage: {confidence.get('feature_coverage', 0):.1f}% (target: 85%+)")
            self.console.print(f"  - Imputed features: {confidence.get('num_imputed', 0)} (target: <3)")
            self.console.print(f"  - Recommendation: Predictions may be less reliable due to limited data")
    
    def export_predictions_table(self, predictions: pd.DataFrame, 
                                filename: str = None) -> str:
        """
        Export predictions to a CSV file
        
        Args:
            predictions: DataFrame with predictions
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        from pathlib import Path
        
        # Create export directory
        export_dir = Path(config.EXPORT_DIR)
        export_dir.mkdir(exist_ok=True)
        
        # Generate filename
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"predictions_{timestamp}.csv"
        
        filepath = export_dir / filename
        
        # Export
        predictions.to_csv(filepath, index=False)
        
        return str(filepath)


if __name__ == "__main__":
    # Test display
    display = F1Display()
    display.show_banner()
    
    # Sample data
    sample_predictions = pd.DataFrame({
        'Driver': ['VER', 'HAM', 'LEC', 'SAI', 'NOR'],
        'Team': ['Red Bull Racing', 'Mercedes', 'Ferrari', 'Ferrari', 'McLaren'],
        'WinProbability': [0.45, 0.25, 0.15, 0.08, 0.05],
        'PodiumProbability': [0.85, 0.75, 0.68, 0.45, 0.35]
    })
    
    display.show_race_header("Monaco Grand Prix", "Circuit de Monaco", "2024-05-26")
    display.show_predictions_table(sample_predictions)
    
    sample_insights = {
        'predicted_winner': {
            'driver': 'VER',
            'team': 'Red Bull Racing',
            'probability': 0.45
        },
        'predicted_podium': [
            {'driver': 'VER', 'team': 'Red Bull Racing', 'probability': 0.85},
            {'driver': 'HAM', 'team': 'Mercedes', 'probability': 0.75},
            {'driver': 'LEC', 'team': 'Ferrari', 'probability': 0.68}
        ]
    }
    
    display.show_winner_prediction(sample_insights)
    display.show_podium_prediction(sample_insights)
