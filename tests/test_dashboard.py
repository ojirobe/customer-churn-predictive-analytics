from streamlit.testing.v1 import AppTest


def test_dashboard_smoke() -> None:
    app = AppTest.from_file("app.py", default_timeout=30)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Customer Churn & Predictive Analytics"
    assert len(app.tabs) == 5

