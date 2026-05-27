"""
🚀 Production Validation Runner
Quick script to run production validation tests
"""

import logging
import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import production validator
from test_production_validation import production_validator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run production validation"""
    logger.info("🚀 Starting AutoForge Production Validation")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run stress tests
        logger.info("\n🧪 RUNNING STRESS TESTS")
        logger.info("-" * 40)
        stress_results = production_validator.run_stress_tests()
        
        # Run performance benchmarks
        logger.info("\n⚡ RUNNING PERFORMANCE BENCHMARKS")
        logger.info("-" * 40)
        performance_results = production_validator.run_performance_benchmarks()
        
        # Run reliability tests
        logger.info("\n🛡️ RUNNING RELIABILITY TESTS")
        logger.info("-" * 40)
        reliability_results = production_validator.run_reliability_tests()
        
        # Get summary
        logger.info("\n📊 GENERATING VALIDATION SUMMARY")
        logger.info("-" * 40)
        summary = production_validator.get_validation_summary()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("🎯 PRODUCTION VALIDATION RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"⏱️  Total execution time: {total_time:.2f} seconds")
        logger.info(f"📈 Overall status: {summary['overall_status'].upper()}")
        logger.info(f"✅ Tests passed: {summary['total_passed']}/{summary['total_tests']}")
        logger.info(f"📊 Pass rate: {summary['pass_rate']:.1%}")
        
        logger.info(f"\n🧪 Stress Tests: {summary['stress_tests_passed']}/5 passed")
        logger.info(f"⚡ Performance Tests: {summary['performance_tests_passed']}/4 passed")
        logger.info(f"🛡️ Reliability Tests: {summary['reliability_tests_passed']}/4 passed")
        
        # Detailed results
        logger.info("\n📋 DETAILED RESULTS:")
        logger.info("-" * 30)
        
        # Stress test details
        if 'stress_tests' in summary['details']:
            logger.info("\n🧪 Stress Tests:")
            for test_name, result in summary['details']['stress_tests'].items():
                status_icon = "✅" if result.get('status') == 'passed' else "❌"
                logger.info(f"  {status_icon} {test_name}: {result.get('status', 'unknown')}")
                if 'execution_time' in result:
                    logger.info(f"     ⏱️  Time: {result['execution_time']:.2f}s")
                if 'error' in result:
                    logger.info(f"     ❌ Error: {result['error']}")
        
        # Performance test details
        if 'performance_tests' in summary['details']:
            logger.info("\n⚡ Performance Tests:")
            for test_name, result in summary['details']['performance_tests'].items():
                status_icon = "✅" if result.get('status') == 'passed' else "❌"
                logger.info(f"  {status_icon} {test_name}: {result.get('status', 'unknown')}")
                if 'avg_samples_per_second' in result:
                    logger.info(f"     📊 Speed: {result['avg_samples_per_second']:.1f} samples/sec")
                if 'error' in result:
                    logger.info(f"     ❌ Error: {result['error']}")
        
        # Reliability test details
        if 'reliability_tests' in summary['details']:
            logger.info("\n🛡️ Reliability Tests:")
            for test_name, result in summary['details']['reliability_tests'].items():
                status_icon = "✅" if result.get('status') == 'passed' else "❌"
                logger.info(f"  {status_icon} {test_name}: {result.get('status', 'unknown')}")
                if 'error' in result:
                    logger.info(f"     ❌ Error: {result['error']}")
        
        # Final verdict
        logger.info("\n" + "=" * 60)
        if summary['overall_status'] == 'passed':
            logger.info("🎉 AUTOFORGE PRODUCTION VALIDATION: PASSED")
            logger.info("✅ System is ready for production deployment!")
        else:
            logger.info("⚠️  AUTOFORGE PRODUCTION VALIDATION: FAILED")
            logger.info("❌ System needs fixes before production deployment")
        
        logger.info("=" * 60)
        
        return summary['overall_status'] == 'passed'
        
    except Exception as e:
        logger.error(f"❌ Production validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
