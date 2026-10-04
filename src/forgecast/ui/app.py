"""ForgeCast Streamlit entrypoint."""

from forgecast.ui.views import configure_page, render_overview


def main() -> None:
    configure_page("Overview")
    render_overview()


if __name__ == "__main__":
    main()
