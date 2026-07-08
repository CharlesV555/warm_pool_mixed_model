import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from polymer_sim.recording.summary import load


def test_dt_compare_averages_simulation_time_by_blended_dt_pair():
    metadata_path = Path("tests") / "_summary_dt_compare_metadata_tmp.json"
    output_dir = Path("tests") / "_summary_dt_compare_output_tmp"
    fig = None
    try:
        metadata_path.write_text(
            json.dumps(
                {
                    "shared": {},
                    "runs": [
                        {"mode": "ssa", "simulation_final_time": 99.0},
                        {
                            "mode": "blended",
                            "simulation_final_time": 1.0,
                            "blended_dt_cle": 0.001,
                            "blended_dt_macro": 0.001,
                        },
                        {
                            "mode": "blended",
                            "simulation_final_time": 3.0,
                            "blended_dt_cle": 0.001,
                            "blended_dt_macro": 0.001,
                        },
                        {
                            "mode": "blended",
                            "simulation_final_time": 4.0,
                            "blended_dt_cle": 0.001,
                            "blended_dt_macro": 0.01,
                        },
                        {
                            "mode": "blended",
                            "simulation_final_time": 8.0,
                            "blended_dt_cle": 0.01,
                            "blended_dt_macro": 0.01,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        batch = load(metadata_path)
        fig, ax, payload = batch.dt_compare(annotate=False, output_dir=output_dir)

        assert ax.get_xlabel() == "dt_macro"
        assert payload["dt_cle_values"] == [0.001, 0.01]
        assert payload["dt_macro_values"] == [0.001, 0.01]
        np.testing.assert_allclose(payload["mean_simulation_time"][0, 0], 2.0)
        np.testing.assert_allclose(payload["mean_simulation_time"][0, 1], 4.0)
        assert np.isnan(payload["mean_simulation_time"][1, 0])
        np.testing.assert_allclose(payload["mean_simulation_time"][1, 1], 8.0)
        np.testing.assert_array_equal(payload["count"], [[2, 1], [0, 1]])
        assert Path(payload["figure_path"]).exists()
        assert Path(payload["table_path"]).exists()
        table_text = Path(payload["table_path"]).read_text(encoding="utf-8")
        assert "dt_cle,dt_macro,mean_simulation_time,n_runs,run_indices" in table_text
        assert "0.001,0.001,2,2" in table_text
    finally:
        if fig is not None:
            plt.close(fig)
        if metadata_path.exists():
            metadata_path.unlink()
        figure_path = output_dir / "dt_compare_heatmap.png"
        table_path = output_dir / "dt_compare_report.csv"
        if figure_path.exists():
            figure_path.unlink()
        if table_path.exists():
            table_path.unlink()
        if output_dir.exists():
            output_dir.rmdir()
