"""
💻 CLI Integration
Connects AutoForge with existing CLI modules
"""

import logging
import argparse
import sys
import os
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    from . import CLICommands, original_main
except ImportError:
    CLICommands = None
    original_main = None

# Import AutoForge components
try:
    from ..core.unified_automl import UnifiedAutoML
    from ..input_output.input_types import AutoMLInput
except ImportError:
    from core.unified_automl import UnifiedAutoML
    from input_output.input_types import AutoMLInput

logger = logging.getLogger(__name__)


class CLIIntegrator:
    """
    Integration layer for CLI functionality
    """
    
    def __init__(self):
        """Initialize CLI integrator"""
        self.available_commands = self._check_available_commands()
        self.command_history = []
        
    def _check_available_commands(self) -> Dict[str, bool]:
        """Check which CLI commands are available"""
        commands = {
            'original_cli': original_main is not None,
            'cli_commands': CLICommands is not None,
            'autoforge_cli': True  # Always available
        }
        
        available_count = sum(commands.values())
        logger.info(f"💻 Available CLI commands: {available_count}/{len(commands)}")
        
        return commands
    
    def create_autoforge_cli(self) -> argparse.ArgumentParser:
        """Create AutoForge CLI with enhanced commands"""
        try:
            parser = argparse.ArgumentParser(
                description='AutoForge - Unified AutoML System',
                formatter_class=argparse.RawDescriptionHelpFormatter,
                epilog="""
Examples:
  autoforge train --data data.csv --target price
  autoforge predict --model model.pkl --data test.csv
  autoforge benchmark --data data.csv --target price
  autoforge info
  autoforge test
                """
            )
            
            # Main subcommands
            subparsers = parser.add_subparsers(dest='command', help='Available commands')
            
            # Train command
            train_parser = subparsers.add_parser('train', help='Train AutoML model')
            train_parser.add_argument('--data', required=True, help='Training data file')
            train_parser.add_argument('--target', required=True, help='Target column name')
            train_parser.add_argument('--output', help='Output model file')
            train_parser.add_argument('--time', type=int, help='Maximum training time in seconds')
            train_parser.add_argument('--trials', type=int, help='Maximum number of trials')
            train_parser.add_argument('--preference', choices=['fast', 'balanced', 'accurate'], 
                                     default='balanced', help='Training preference')
            train_parser.add_argument('--no-tracking', action='store_true', help='Disable experiment tracking')
            train_parser.add_argument('--no-monitoring', action='store_true', help='Disable performance monitoring')
            train_parser.add_argument('--no-optimization', action='store_true', help='Disable hyperparameter optimization')
            
            # Predict command
            predict_parser = subparsers.add_parser('predict', help='Make predictions')
            predict_parser.add_argument('--model', required=True, help='Trained model file')
            predict_parser.add_argument('--data', required=True, help='Data file for prediction')
            predict_parser.add_argument('--output', help='Output file for predictions')
            
            # Benchmark command
            benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark AutoML performance')
            benchmark_parser.add_argument('--data', required=True, help='Data file for benchmarking')
            benchmark_parser.add_argument('--target', required=True, help='Target column name')
            benchmark_parser.add_argument('--iterations', type=int, default=5, help='Number of benchmark iterations')
            benchmark_parser.add_argument('--compare', action='store_true', help='Compare with existing systems')
            
            # Info command
            info_parser = subparsers.add_parser('info', help='Show system information')
            info_parser.add_argument('--detailed', action='store_true', help='Show detailed information')
            
            # Test command
            test_parser = subparsers.add_parser('test', help='Run system tests')
            test_parser.add_argument('--quick', action='store_true', help='Run quick tests only')
            test_parser.add_argument('--integration', action='store_true', help='Run integration tests')
            
            # Integration command
            integration_parser = subparsers.add_parser('integrate', help='Run full integration test')
            integration_parser.add_argument('--data', help='Data file for integration test')
            integration_parser.add_argument('--target', help='Target column name')
            
            return parser
            
        except Exception as e:
            logger.error(f"❌ CLI creation failed: {e}")
            return self._create_fallback_cli()
    
    def _create_fallback_cli(self) -> argparse.ArgumentParser:
        """Create fallback CLI"""
        parser = argparse.ArgumentParser(description='AutoForge - Unified AutoML System')
        parser.add_argument('command', choices=['train', 'predict', 'info', 'test'], help='Command to run')
        parser.add_argument('--data', help='Data file')
        parser.add_argument('--target', help='Target column')
        parser.add_argument('--model', help='Model file')
        return parser
    
    def run_train_command(self, args) -> Dict[str, Any]:
        """Run train command"""
        try:
            logger.info("🚀 Starting AutoForge training...")
            
            # Load data
            data = pd.read_csv(args.data)
            if args.target not in data.columns:
                raise ValueError(f"Target column '{args.target}' not found in data")
            
            # Create AutoML input
            automl_input = AutoMLInput(
                data=data,
                target_column=args.target,
                max_time=args.time,
                max_trials=args.trials,
                user_preference=args.preference
            )
            
            # Initialize AutoForge
            autoforge = UnifiedAutoML()
            
            # Train model
            autoforge.fit(
                automl_input,
                enable_tracking=not args.no_tracking,
                enable_monitoring=not args.no_monitoring,
                enable_optimization=not args.no_optimization
            )
            
            # Save model
            output_path = args.output or f"autoforge_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            autoforge.save_model(output_path)
            
            # Get performance stats
            stats = autoforge.get_performance_stats()
            
            result = {
                'status': 'success',
                'model_saved': output_path,
                'performance': stats,
                'message': f"Model trained and saved to {output_path}"
            }
            
            # Store command history
            self.command_history.append({
                'command': 'train',
                'args': vars(args),
                'result': result,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Training complete: {output_path}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_predict_command(self, args) -> Dict[str, Any]:
        """Run predict command"""
        try:
            logger.info("🔮 Starting AutoForge prediction...")
            
            # Load model
            autoforge = UnifiedAutoML()
            autoforge.load_model(args.model)
            
            # Load data
            data = pd.read_csv(args.data)
            
            # Make predictions
            predictions = autoforge.predict(data)
            
            # Save predictions
            if args.output:
                output_df = data.copy()
                output_df['predictions'] = predictions
                output_df.to_csv(args.output, index=False)
                result = {
                    'status': 'success',
                    'predictions_saved': args.output,
                    'n_predictions': len(predictions),
                    'message': f"Predictions saved to {args.output}"
                }
            else:
                result = {
                    'status': 'success',
                    'predictions': predictions.tolist(),
                    'n_predictions': len(predictions),
                    'message': f"Generated {len(predictions)} predictions"
                }
            
            # Store command history
            self.command_history.append({
                'command': 'predict',
                'args': vars(args),
                'result': result,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Prediction complete: {len(predictions)} predictions")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_benchmark_command(self, args) -> Dict[str, Any]:
        """Run benchmark command"""
        try:
            logger.info("📊 Starting AutoForge benchmark...")
            
            # Load data
            data = pd.read_csv(args.data)
            if args.target not in data.columns:
                raise ValueError(f"Target column '{args.target}' not found in data")
            
            # Initialize AutoForge
            autoforge = UnifiedAutoML()
            
            # Create test datasets
            from sklearn.model_selection import train_test_split
            X = data.drop(columns=[args.target])
            y = data[args.target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            results = []
            for i in range(args.iterations):
                logger.info(f"📊 Benchmark iteration {i+1}/{args.iterations}")
                
                # Train model
                automl_input = AutoMLInput(
                    data=pd.concat([X_train, y_train.rename('target')], axis=1),
                    target_column='target',
                    max_trials=20  # Reduced for benchmarking
                )
                
                autoforge.fit(automl_input, enable_tracking=False, enable_monitoring=False, enable_optimization=False)
                
                # Evaluate
                predictions = autoforge.predict(X_test)
                from sklearn.metrics import accuracy_score, r2_score
                
                if len(np.unique(y_test)) < 20:  # Classification
                    score = accuracy_score(y_test, predictions)
                else:  # Regression
                    score = r2_score(y_test, predictions)
                
                results.append(score)
            
            # Calculate statistics
            avg_score = sum(results) / len(results)
            best_score = max(results)
            worst_score = min(results)
            
            benchmark_result = {
                'status': 'success',
                'iterations': args.iterations,
                'average_score': avg_score,
                'best_score': best_score,
                'worst_score': worst_score,
                'all_scores': results,
                'message': f"Benchmark complete: {avg_score:.4f} average score"
            }
            
            # Store command history
            self.command_history.append({
                'command': 'benchmark',
                'args': vars(args),
                'result': benchmark_result,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            logger.info(f"✅ Benchmark complete: {avg_score:.4f} average score")
            return benchmark_result
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_info_command(self, args) -> Dict[str, Any]:
        """Run info command"""
        try:
            autoforge = UnifiedAutoML()
            
            # Get integration summary
            integration_summary = autoforge.get_integration_summary()
            
            info_result = {
                'status': 'success',
                'autoforge_version': '1.0.0',
                'integration_summary': integration_summary,
                'available_components': {
                    'api_systems': integration_summary['api_integrator']['total_systems'],
                    'benchmarkers': integration_summary['benchmarking_integrator']['total_benchmarkers'],
                    'systemization': integration_summary['systemization_integrator']['total_components'],
                    'tracking': integration_summary['tracking_integrator']['total_components'],
                    'optimizers': integration_summary['optimizer_integrator']['total_optimizers'],
                    'features': integration_summary['feature_registry']['total_features'],
                    'models': integration_summary['model_registry']['total_models']
                }
            }
            
            if args.detailed:
                info_result['detailed_info'] = integration_summary
            
            # Store command history
            self.command_history.append({
                'command': 'info',
                'args': vars(args),
                'result': info_result,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            return info_result
            
        except Exception as e:
            logger.error(f"❌ Info command failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_test_command(self, args) -> Dict[str, Any]:
        """Run test command"""
        try:
            logger.info("🧪 Running AutoForge tests...")
            
            # Import test functions
            from ..tests.test_unified_system import TestUnifiedAutoML
            
            # Run tests
            test_suite = TestUnifiedAutoML()
            
            if args.quick:
                results = test_suite.test_basic_workflow()
            elif args.integration:
                results = test_suite.test_integration()
            else:
                results = test_suite.run_all_tests()
            
            test_result = {
                'status': 'success',
                'test_results': results,
                'message': f"Tests completed: {results.get('passed', 0)}/{results.get('total', 0)} passed"
            }
            
            # Store command history
            self.command_history.append({
                'command': 'test',
                'args': vars(args),
                'result': test_result,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            return test_result
            
        except Exception as e:
            logger.error(f"❌ Test command failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_integration_command(self, args) -> Dict[str, Any]:
        """Run integration command"""
        try:
            logger.info("🔗 Running full integration test...")
            
            if args.data and args.target:
                # Use provided data
                data = pd.read_csv(args.data)
                X = data.drop(columns=[args.target])
                y = data[args.target]
            else:
                # Use sample data
                from ..utils.helpers import create_sample_data
                X, y = create_sample_data(n_samples=1000, n_features=10)
            
            # Initialize AutoForge
            autoforge = UnifiedAutoML()
            
            # Run full integration
            integration_results = autoforge.run_full_integration(X, y)
            
            integration_result = {
                'status': 'success',
                'integration_results': integration_results,
                'message': "Full integration test completed"
            }
            
            # Store command history
            self.command_history.append({
                'command': 'integrate',
                'args': vars(args),
                'result': integration_result,
                'timestamp': pd.Timestamp.now().isoformat()
            })
            
            return integration_result
            
        except Exception as e:
            logger.error(f"❌ Integration command failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def get_command_history(self) -> List[Dict[str, Any]]:
        """Get command history"""
        return self.command_history
    
    def generate_cli_report(self) -> str:
        """Generate CLI report"""
        lines = []
        
        # Header
        lines.append("💻 AutoForge CLI Report")
        lines.append("=" * 50)
        
        # Command availability
        lines.append("\n📋 Available Commands:")
        for command, available in self.available_commands.items():
            status = "✅" if available else "❌"
            lines.append(f"  {status} {command}")
        
        # Command history
        if self.command_history:
            lines.append(f"\n🎯 Command History:")
            lines.append(f"  Total Commands: {len(self.command_history)}")
            
            command_counts = {}
            for cmd in self.command_history:
                cmd_name = cmd['command']
                command_counts[cmd_name] = command_counts.get(cmd_name, 0) + 1
            
            for command, count in command_counts.items():
                lines.append(f"  {command}: {count} times")
        
        return "\n".join(lines)


# Global CLI integrator instance
cli_integrator = CLIIntegrator()


def create_autoforge_cli():
    """Create AutoForge CLI"""
    return cli_integrator.create_autoforge_cli()


def main():
    """Main CLI entry point"""
    try:
        parser = create_autoforge_cli()
        args = parser.parse_args()
        
        if args.command == 'train':
            result = cli_integrator.run_train_command(args)
        elif args.command == 'predict':
            result = cli_integrator.run_predict_command(args)
        elif args.command == 'benchmark':
            result = cli_integrator.run_benchmark_command(args)
        elif args.command == 'info':
            result = cli_integrator.run_info_command(args)
        elif args.command == 'test':
            result = cli_integrator.run_test_command(args)
        elif args.command == 'integrate':
            result = cli_integrator.run_integration_command(args)
        else:
            parser.print_help()
            return
        
        # Print result
        if result['status'] == 'success':
            print(f"✅ {result['message']}")
        else:
            print(f"❌ {result['message']}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ CLI failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
