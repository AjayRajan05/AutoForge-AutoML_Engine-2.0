"""
🚀 AutoForge Main Entry Point
Command-line interface and main execution
"""

import argparse
import sys
import logging
import pandas as pd
from pathlib import Path

try:
    from .core.unified_automl import UnifiedAutoML
    from .input_output.input_types import AutoMLInput, get_test_input
    from .input_output.input_validator import InputValidator
except ImportError:
    from core.unified_automl import UnifiedAutoML
    from input_output.input_types import AutoMLInput, get_test_input
    from input_output.input_validator import InputValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='AutoForge - Intelligent AutoML System')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train AutoML model')
    train_parser.add_argument('--data', required=True, help='Path to training data CSV file')
    train_parser.add_argument('--target', required=True, help='Target column name')
    train_parser.add_argument('--task-type', choices=['classification', 'regression'], help='Task type')
    train_parser.add_argument('--data-type', choices=['tabular', 'text', 'time_series', 'image'], help='Data type')
    train_parser.add_argument('--model-family', choices=['ml', 'dl', 'both'], default='ml',
                             help='Model family: traditional ML, deep learning, or both')
    train_parser.add_argument('--search-depth', choices=['fast', 'balanced', 'deep'],
                             default='balanced', help='Hyperparameter search depth')
    train_parser.add_argument('--preference', choices=['auto', 'fast', 'accurate', 'robust'], 
                             default='auto', help='User preference')
    train_parser.add_argument('--max-time', type=float, help='Maximum training time (seconds)')
    train_parser.add_argument('--max-trials', type=int, help='Maximum optimization trials')
    train_parser.add_argument('--save-model', help='Save model with this name')
    train_parser.add_argument('--report', action='store_true',
                             help='Print training report path after save')
    train_parser.add_argument('--output', help='Output file for results')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', required=True, help='Model name or path')
    predict_parser.add_argument('--data', required=True, help='Path to prediction data CSV file')
    predict_parser.add_argument('--output', required=True, help='Output file for predictions')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test examples')
    test_parser.add_argument('--type', choices=['small', 'medium', 'large', 'text', 'time_series'], 
                            default='small', help='Test dataset type')
    test_parser.add_argument('--save-model', help='Save trained model')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get information about saved models')
    info_parser.add_argument('--model', help='Specific model name to get info about')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'predict':
            predict_command(args)
        elif args.command == 'test':
            test_command(args)
        elif args.command == 'info':
            info_command(args)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


def train_command(args):
    """Handle train command"""
    logger.info(f"🚀 Starting AutoML training on {args.data}")
    
    # Load data
    try:
        data = pd.read_csv(args.data)
        logger.info(f"📊 Loaded data: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Create AutoML input
    automl_input = AutoMLInput(
        data=data,
        target_column=args.target,
        task_type=args.task_type,
        data_type=args.data_type,
        model_family=args.model_family,
        user_preference=args.preference,
        search_depth=args.search_depth,
        max_time=args.max_time,
        max_trials=args.max_trials
    )

    automl = UnifiedAutoML({
        'model_family': args.model_family,
        'search_depth': args.search_depth,
    })
    automl.fit(automl_input)
    
    # Get results
    performance_stats = automl.get_performance_stats()
    explanations = automl.explain()
    
    # Save model if requested
    if args.save_model:
        model_dir = automl.save_model(args.save_model)
        logger.info(f"💾 Model saved as: {args.save_model}")
        if args.report:
            report_path = Path(model_dir) / "REPORT.md"
            print(f"📄 Training report: {report_path}")
            print(automl.selection_summary())
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            f.write("AutoForge AutoML Training Results\n")
            f.write("=" * 40 + "\n")
            f.write(f"Best Score: {performance_stats['best_score']:.4f}\n")
            f.write(f"Training Time: {performance_stats['training_time']:.2f}s\n")
            f.write(f"Model Type: {performance_stats['model_type']}\n")
            f.write(f"Features Used: {performance_stats['n_features']}\n")
            f.write(f"Task Type: {performance_stats['task_type']}\n")
            f.write(f"Data Type: {performance_stats['data_type']}\n")
            f.write(f"Model Family: {performance_stats.get('model_family', 'ml')}\n")
            f.write(f"Strategy: {performance_stats['strategy']}\n\n")
            f.write("Model Comparison\n")
            f.write("-" * 40 + "\n")
            f.write(automl.print_model_comparison())
        logger.info(f"📄 Results saved to: {args.output}")
    
    # Print summary
    print("\n🎉 AutoForge Training Complete!")
    print(f"📊 Best Score: {performance_stats['best_score']:.4f}")
    print(f"⏱️  Training Time: {performance_stats['training_time']:.2f}s")
    print(f"🤖 Model Type: {performance_stats['model_type']}")
    print(f"🔧 Features Used: {performance_stats['n_features']}")
    print(f"🎯 Strategy: {performance_stats['strategy']}")


def predict_command(args):
    """Handle predict command"""
    logger.info(f"🔮 Making predictions with model {args.model}")
    
    # Load model
    automl = UnifiedAutoML()
    automl.load_model(args.model)
    
    # Load data
    try:
        data = pd.read_csv(args.data)
        logger.info(f"📊 Loaded prediction data: {data.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return
    
    # Make predictions
    predictions = automl.predict(data)
    
    # Save predictions
    results_df = data.copy()
    results_df['predictions'] = predictions
    results_df.to_csv(args.output, index=False)
    
    logger.info(f"💾 Predictions saved to: {args.output}")
    print(f"✅ Predictions saved for {len(predictions)} samples")


def test_command(args):
    """Handle test command"""
    logger.info(f"🧪 Running AutoForge test with {args.type} dataset")
    
    # Get test data
    if args.type == 'small':
        test_input = get_test_input()
        description = "small synthetic dataset"
    elif args.type == 'medium':
        from .input_output.input_types import get_large_test_input
        test_input = get_large_test_input()
        description = "large synthetic dataset"
    elif args.type == 'text':
        from .input_output.input_types import get_text_test_input
        test_input = get_text_test_input()
        description = "text dataset"
    elif args.type == 'time_series':
        from .input_output.input_types import get_time_series_test_input
        test_input = get_time_series_test_input()
        description = "time series dataset"
    else:
        test_input = get_test_input()
        description = "default synthetic dataset"
    
    print(f"\n🧪 Testing AutoForge with {description}")
    print(f"📊 Dataset shape: {test_input.get_shape()}")
    print(f"🎯 Task type: {test_input.task_type}")
    print(f"🔬 Data type: {test_input.data_type}")
    
    # Initialize and train AutoML
    automl = UnifiedAutoML()
    automl.fit(test_input)
    
    # Get results
    performance_stats = automl.get_performance_stats()
    explanations = automl.explain()
    
    # Save model if requested
    if args.save_model:
        automl.save_model(args.save_model)
        logger.info(f"💾 Model saved as: {args.save_model}")
    
    # Make some test predictions
    X_test = test_input.get_features().head(10)
    predictions = automl.predict(X_test)
    
    # Print results
    print(f"\n🎉 Test Complete!")
    print(f"📊 Best Score: {performance_stats['best_score']:.4f}")
    print(f"⏱️  Training Time: {performance_stats['training_time']:.2f}s")
    print(f"🤖 Model Type: {performance_stats['model_type']}")
    print(f"🔧 Features Used: {performance_stats['n_features']}")
    print(f"🎯 Task Type: {performance_stats['task_type']}")
    print(f"🔬 Data Type: {performance_stats['data_type']}")
    print(f"🧠 Strategy: {performance_stats['strategy']}")
    
    print(f"\n🔮 Sample Predictions (first 10):")
    for i, pred in enumerate(predictions[:5]):
        print(f"  Sample {i+1}: {pred}")
    
    # Feature importance if available
    feature_importance = automl.get_feature_importance()
    if feature_importance:
        print(f"\n🔍 Top 5 Feature Importance:")
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")


def info_command(args):
    """Handle info command"""
    from .persistence.model_saver import ModelSaver
    
    model_saver = ModelSaver()
    
    if args.model:
        # Get info about specific model
        model_info = model_saver.get_model_info(args.model)
        if model_info:
            print(f"\n📋 Model Information: {args.model}")
            print(f"🤖 Model Type: {model_info.get('model_type', 'Unknown')}")
            print(f"💾 Save Timestamp: {model_info.get('save_timestamp', 'Unknown')}")
            print(f"📁 File Path: {model_info.get('file_path', 'Unknown')}")
            print(f"📏 File Size: {model_info.get('file_size_bytes', 0):,} bytes")
            print(f"🎯 Format: {model_info.get('format', 'Unknown')}")
        else:
            print(f"❌ Model '{args.model}' not found")
    else:
        # List all models
        models = model_saver.list_saved_models()
        if models:
            print(f"\n📋 Saved Models ({len(models)}):")
            for name, info in models.items():
                print(f"  📦 {name}")
                print(f"     Type: {info.get('model_type', 'Unknown')}")
                print(f"     Saved: {info.get('save_timestamp', 'Unknown')}")
                print(f"     Score: {info.get('best_score', 'N/A')}")
                print()
        else:
            print("📭 No saved models found")


if __name__ == '__main__':
    main()
