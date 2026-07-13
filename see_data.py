"""Interactive Streamlit viewer for the CSV files generated in ``output/``.

Run with: streamlit run see_data.py
"""

from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st


OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def available_csvs() -> list[Path]:
    """Return every CSV currently present directly in the output directory."""
    if not OUTPUT_DIR.exists():
        return []
    return sorted(OUTPUT_DIR.glob("*.csv"), key=lambda path: path.name.lower())


@st.cache_data(show_spinner="Loading CSV...")
def load_csv(path_as_string: str) -> pd.DataFrame:
    """Load a generated CSV, preserving its first saved index column."""
    return pd.read_csv(path_as_string, index_col=0)


def selected_file_from_url(filenames: list[str]) -> str | None:
    """Get a valid selected file from ``?file=...`` for new-tab links."""
    requested_file = st.query_params.get("file")
    if isinstance(requested_file, list):
        requested_file = requested_file[0] if requested_file else None
    return requested_file if requested_file in filenames else None


def show_dataframe(file_path: Path, widget_scope: str) -> None:
    """Show summary information and the contents of one CSV file."""
    try:
        dataframe = load_csv(str(file_path))
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read `{file_path.name}`: {error}")
        return

    rows, columns = dataframe.shape
    first, second = st.columns(2)
    first.metric("Rows", f"{rows:,}")
    second.metric("Columns", columns)
    st.dataframe(dataframe, use_container_width=True, height=600)
    st.download_button(
        "Download selected CSV",
        data=file_path.read_bytes(),
        file_name=file_path.name,
        mime="text/csv",
        key=f"download-{widget_scope}-{file_path.name}",
    )


def main() -> None:
    st.set_page_config(page_title="Output CSV Viewer", layout="wide")
    st.title("Output CSV Viewer")
    st.caption("Browse output files, open one in another browser tab, or compare two files.")

    csv_paths = available_csvs()
    if not csv_paths:
        st.warning(f"No CSV files were found in `{OUTPUT_DIR}`.")
        return

    filenames = [path.name for path in csv_paths]
    file_paths = {path.name: path for path in csv_paths}
    linked_file = selected_file_from_url(filenames)

    st.success(f"{len(filenames)} CSV files found in `output/`.")

    single_tab, compare_tab = st.tabs(["View one CSV", "Compare two CSVs"])

    with single_tab:
        default_index = filenames.index(linked_file) if linked_file else 0
        selected_name = st.selectbox(
            "Choose a CSV file",
            filenames,
            index=default_index,
            key="single-file-selector",
        )
        st.caption(f"Viewing `output/{selected_name}`")

        # target=_blank creates an independent Streamlit browser session, initialized
        # with the selected filename through the query parameter.
        new_tab_url = f"?file={quote(selected_name)}"
        st.markdown(
            f'<a href="{new_tab_url}" target="_blank" rel="noopener noreferrer">'
            "Open this CSV in a new browser tab ↗</a>",
            unsafe_allow_html=True,
        )
        show_dataframe(file_paths[selected_name], "single")

    with compare_tab:
        st.caption("Choose two files to display independently, side by side.")
        selector_left, selector_right = st.columns(2)
        with selector_left:
            left_name = st.selectbox("Left CSV", filenames, key="left-file-selector")
        with selector_right:
            right_name = st.selectbox(
                "Right CSV",
                filenames,
                index=1 if len(filenames) > 1 else 0,
                key="right-file-selector",
            )

        if left_name == right_name:
            st.info("Both sides show the same file. Choose another CSV to compare them.")
        left_column, right_column = st.columns(2)
        with left_column:
            st.subheader(left_name)
            show_dataframe(file_paths[left_name], "compare-left")
        with right_column:
            st.subheader(right_name)
            show_dataframe(file_paths[right_name], "compare-right")


if __name__ == "__main__":
    main()