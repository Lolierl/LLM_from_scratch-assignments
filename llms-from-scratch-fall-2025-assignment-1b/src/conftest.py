def pytest_report_teststatus(report, config):
    if report.passed:
        return report.outcome, "", ""
    return None
