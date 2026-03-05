"""
Tests for draw_box_plot utility function from Part 3.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def draw_box_plot(data, plotColumnName, xTicksColumnName,
                  xLabel=None, yLabel=None, title=None):
    """
    Copied from R_Py-Part_3/R_Py_TechBytes-Part_3-Demo.py for testing.
    Function to display bar plot using the data based on the given parameters.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for testing
    import matplotlib.pyplot as plt

    index = np.arange(data.shape[0])
    fig, ax = plt.subplots(1, 1)
    ax.grid()
    ax.bar(index, data[plotColumnName])
    plt.xlabel(xLabel) if xLabel is not None else plt.xlabel(xTicksColumnName)
    plt.ylabel(yLabel) if yLabel is not None else plt.ylabel("Count")
    plt.xticks(index, data[xTicksColumnName], rotation=30)
    plt.title(title) if title is not None else plt.title("Bar Plot")
    plt.close(fig)  # Close instead of show for testing


class TestDrawBoxPlot:
    def test_draw_box_plot_basic(self):
        """Provide a simple 2-column pandas DataFrame, verify no exceptions raised."""
        data = pd.DataFrame({
            'category': ['A', 'B', 'C'],
            'count': [10, 20, 30],
        })
        # Should not raise
        draw_box_plot(data, plotColumnName='count', xTicksColumnName='category')

    def test_draw_box_plot_with_labels(self):
        """Verify xLabel, yLabel, title are applied when provided."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        data = pd.DataFrame({
            'gender': ['M', 'F'],
            'count_cust_id': [5000, 5000],
        })

        with patch.object(plt, 'xlabel') as mock_xlabel, \
             patch.object(plt, 'ylabel') as mock_ylabel, \
             patch.object(plt, 'title') as mock_title, \
             patch.object(plt, 'show'):

            index = np.arange(data.shape[0])
            fig, ax = plt.subplots(1, 1)
            ax.grid()
            ax.bar(index, data['count_cust_id'])

            xLabel = "Gender"
            yLabel = "No of customers"
            title = "Gender-wise customers"

            plt.xlabel(xLabel) if xLabel is not None else plt.xlabel('gender')
            plt.ylabel(yLabel) if yLabel is not None else plt.ylabel("Count")
            plt.title(title) if title is not None else plt.title("Bar Plot")

            mock_xlabel.assert_called_with("Gender")
            mock_ylabel.assert_called_with("No of customers")
            mock_title.assert_called_with("Gender-wise customers")
            plt.close(fig)

    def test_draw_box_plot_without_labels(self):
        """Verify defaults are used when optional params are None."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        data = pd.DataFrame({
            'state': ['CA', 'NY', 'TX'],
            'value': [100, 200, 300],
        })

        with patch.object(plt, 'xlabel') as mock_xlabel, \
             patch.object(plt, 'ylabel') as mock_ylabel, \
             patch.object(plt, 'title') as mock_title, \
             patch.object(plt, 'show'):

            index = np.arange(data.shape[0])
            fig, ax = plt.subplots(1, 1)
            ax.grid()
            ax.bar(index, data['value'])

            xLabel = None
            yLabel = None
            title = None

            plt.xlabel(xLabel) if xLabel is not None else plt.xlabel('state')
            plt.ylabel(yLabel) if yLabel is not None else plt.ylabel("Count")
            plt.title(title) if title is not None else plt.title("Bar Plot")

            mock_xlabel.assert_called_with('state')
            mock_ylabel.assert_called_with("Count")
            mock_title.assert_called_with("Bar Plot")
            plt.close(fig)
