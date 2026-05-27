"""Smoke tests for systemization integration."""

import pytest


def test_systemization_imports():
    from systemization.systemization_integration import SystemizationIntegrator
    from systemization.ab_testing import ABTestingFramework
    from systemization.lightweight_monitoring import LightweightMonitor

    integrator = SystemizationIntegrator()
    assert integrator.available_components['monitoring'] is True
    assert integrator.available_components['ab_testing'] is True


def test_setup_monitoring_returns_monitor():
    from systemization.systemization_integration import SystemizationIntegrator

    integrator = SystemizationIntegrator()
    monitor = integrator.setup_monitoring(None)
    assert monitor is not None
    assert hasattr(monitor, 'log_prediction')


def test_monitor_summary():
    from systemization.lightweight_monitoring import LightweightMonitor

    monitor = LightweightMonitor(monitor_dir="monitoring_test")
    summary = monitor.get_monitoring_summary()
    assert isinstance(summary, dict)
