import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

import matplotlib.pyplot as plt
import numpy
import matplotlib.pyplot
from unittest import mock
import unittest
import random
import string
from csep.utils import plots
import csep
from csep.core.catalogs import CSEPCatalog


class TestPoissonPlots(unittest.TestCase):

    def test_SingleNTestPlot(self):

        expected_val = numpy.random.randint(0, 20)
        observed_val = numpy.random.randint(0, 20)
        Ntest_result = mock.Mock()
        Ntest_result.name = 'Mock NTest'
        Ntest_result.sim_name = 'Mock SimName'
        Ntest_result.test_distribution = ['poisson', expected_val]
        Ntest_result.observed_statistic = observed_val
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Ntest_result)

        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in [Ntest_result]])
        self.assertEqual(matplotlib.pyplot.gca().get_title(),
                         Ntest_result.name)

    def test_MultiNTestPlot(self, show=False):

        n_plots = numpy.random.randint(1, 20)
        Ntests = []
        for n in range(n_plots):
            Ntest_result = mock.Mock()
            Ntest_result.name = 'Mock NTest'
            Ntest_result.sim_name = ''.join(
                random.choice(string.ascii_letters) for _ in range(8))
            Ntest_result.test_distribution = ['poisson',
                                              numpy.random.randint(0, 20)]
            Ntest_result.observed_statistic = numpy.random.randint(0, 20)
            Ntests.append(Ntest_result)
        matplotlib.pyplot.close()
        ax = plots.plot_poisson_consistency_test(Ntests)
        Ntests.reverse()

        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in Ntests])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Ntests[0].name)
        if show:
            matplotlib.pyplot.show()

    def test_MultiSTestPlot(self, show=False):

        s_plots = numpy.random.randint(1, 20)
        Stests = []
        for n in range(s_plots):
            Stest_result = mock.Mock()  # Mock class with random attributes
            Stest_result.name = 'Mock STest'
            Stest_result.sim_name = ''.join(
                random.choice(string.ascii_letters) for _ in range(8))
            Stest_result.test_distribution = numpy.random.uniform(-1000, 0,
                                                                  numpy.random.randint(
                                                                      3,
                                                                      500)).tolist()
            Stest_result.observed_statistic = numpy.random.uniform(-1000,
                                                                   0)  # random observed statistic
            if numpy.random.random() < 0.02:  # sim possible infinite values
                Stest_result.observed_statistic = -numpy.inf
            Stests.append(Stest_result)
        matplotlib.pyplot.close()
        plots.plot_poisson_consistency_test(Stests)
        Stests.reverse()
        self.assertEqual(
            [i.get_text() for i in matplotlib.pyplot.gca().get_yticklabels()],
            [i.sim_name for i in Stests])
        self.assertEqual(matplotlib.pyplot.gca().get_title(), Stests[0].name)

    def test_MultiTTestPlot(self, show=False):

        for i in range(10):
            t_plots = numpy.random.randint(2, 20)
            t_tests = []

            def rand(limit=10, offset=0):
                return limit * (numpy.random.random() - offset)

            for n in range(t_plots):
                t_result = mock.Mock()  # Mock class with random attributes
                t_result.name = 'CSEP1 Comparison Test'
                t_result.sim_name = (
                    ''.join(random.choice(string.ascii_letters)
                            for _ in range(8)), 'ref')
                t_result.observed_statistic = rand(offset=0.5)
                t_result.test_distribution = [
                    t_result.observed_statistic - rand(5),
                    t_result.observed_statistic + rand(5)]

                if numpy.random.random() < 0.05:  # sim possible infinite values
                    t_result.observed_statistic = -numpy.inf
                t_tests.append(t_result)
            matplotlib.pyplot.close()
            plots.plot_comparison_test(t_tests)
            t_tests.reverse()
            self.assertEqual(
                [i.get_text() for i in
                 matplotlib.pyplot.gca().get_xticklabels()],
                [i.sim_name[0] for i in t_tests[::-1]])
            self.assertEqual(matplotlib.pyplot.gca().get_title(),
                             t_tests[0].name)


class TestAlarmBasedPlots(unittest.TestCase):
    """Tests for alarm-based evaluation plots (ROC and Molchan diagrams)."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are shared across all test methods."""
        # Load forecast
        forecast_path = 'tests/artifacts/example_csep1_forecasts/Forecast/EEPAS-0F_12_1_2007.dat'
        cls.forecast = csep.load_gridded_forecast(forecast_path, name='EEPAS-0F')

        # Create synthetic catalog with events
        n_events = 15
        bbox = cls.forecast.region.get_bbox()
        min_lon, max_lon = bbox[0], bbox[1]
        min_lat, max_lat = bbox[2], bbox[3]

        # Generate random events within region bounds
        import time
        current_time = int(time.time() * 1000)
        lons = numpy.random.uniform(min_lon, max_lon, n_events)
        lats = numpy.random.uniform(min_lat, max_lat, n_events)
        magnitudes = numpy.random.uniform(4.0, 6.0, n_events)
        depths = numpy.random.uniform(0, 30, n_events)

        # Create catalog data
        catalog_data = [(i, current_time + i*1000, float(lats[i]), float(lons[i]),
                         float(depths[i]), float(magnitudes[i]))
                        for i in range(n_events)]

        cls.catalog = CSEPCatalog(data=catalog_data, region=cls.forecast.region)

    def test_plot_ROC_diagram_linear(self):
        """Test plot_ROC_diagram with linear x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_ROC_diagram(
            self.forecast, self.catalog,
            linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_plot_ROC_diagram_log(self):
        """Test plot_ROC_diagram with logarithmic x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_ROC_diagram(
            self.forecast, self.catalog,
            linear=False, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_plot_Molchan_diagram_linear(self):
        """Test plot_Molchan_diagram with linear x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_Molchan_diagram(
            self.forecast, self.catalog,
            linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_plot_Molchan_diagram_log(self):
        """Test plot_Molchan_diagram with logarithmic x-axis."""
        matplotlib.pyplot.close()
        ax = plots.plot_Molchan_diagram(
            self.forecast, self.catalog,
            linear=False, show=False, savepdf=False, savepng=False
        )
        self.assertIsNotNone(ax)
        matplotlib.pyplot.close()

    def test_ROC_diagram_with_custom_axes(self):
        """Test ROC diagram can use custom axes."""
        matplotlib.pyplot.close()
        fig, ax = matplotlib.pyplot.subplots()
        result_ax = plots.plot_ROC_diagram(
            self.forecast, self.catalog,
            axes=ax, linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertEqual(ax, result_ax)
        matplotlib.pyplot.close()

    def test_Molchan_diagram_with_custom_axes(self):
        """Test Molchan diagram can use custom axes."""
        matplotlib.pyplot.close()
        fig, ax = matplotlib.pyplot.subplots()
        result_ax = plots.plot_Molchan_diagram(
            self.forecast, self.catalog,
            axes=ax, linear=True, show=False, savepdf=False, savepng=False
        )
        self.assertEqual(ax, result_ax)
        matplotlib.pyplot.close()

    def test_ROC_diagram_region_mismatch(self):
        """Test that ROC diagram raises error when regions don't match."""
        # Create catalog with different region
        from csep.core.regions import california_relm_region
        catalog_data = [(0, 1000, 35.0, -120.0, 10.0, 5.0)]
        bad_catalog = CSEPCatalog(data=catalog_data, region=california_relm_region())

        matplotlib.pyplot.close()
        with self.assertRaises(RuntimeError):
            plots.plot_ROC_diagram(
                self.forecast, bad_catalog,
                linear=True, show=False, savepdf=False, savepng=False
            )
        matplotlib.pyplot.close()

    def test_Molchan_diagram_region_mismatch(self):
        """Test that Molchan diagram raises error when regions don't match."""
        # Create catalog with different region
        from csep.core.regions import california_relm_region
        catalog_data = [(0, 1000, 35.0, -120.0, 10.0, 5.0)]
        bad_catalog = CSEPCatalog(data=catalog_data, region=california_relm_region())

        matplotlib.pyplot.close()
        with self.assertRaises(RuntimeError):
            plots.plot_Molchan_diagram(
                self.forecast, bad_catalog,
                linear=True, show=False, savepdf=False, savepng=False
            )
        matplotlib.pyplot.close()


if __name__ == '__main__':
    unittest.main()
