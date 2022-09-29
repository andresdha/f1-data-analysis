import os
from typing import List
import fastf1 as ff1
from fastf1.utils import delta_time
from fastf1 import plotting
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd

# Setup plotting
plotting.setup_mpl()


class QualifyingComparer:
    """Class that runs pipeline to compare the telemetry from the fastest laps of two
    drivers in the qualifying session of the weekend.

        Args:
            drivers (List[str]): List containing the abbreviations of the two drivers
                that are to be compared. First is reference, second compare.
                Example_ ["LEC", "VER"]
            grand_prix (str): Name of the Grand Prix to be queried.
            year (int): Year of the desired Grand Prix.
            path_to_cache (str): Path to directory where cache will be stored. (Defaults
                to '../data_cache')

    """

    def __init__(
        self,
        drivers: List[str],
        grand_prix: str,
        year: int,
        path_to_cache: str = "data_cache/",
    ) -> None:
        """Please see help(QualifyingComparer) for more info"""

        self.driver_1 = drivers[0]
        self.driver_2 = drivers[1]
        self.grand_prix = grand_prix
        self.year = year
        self.cache_path = path_to_cache
        self.session = None
        self.laps_driver_1 = None
        self.laps_driver_2 = None
        self.fastest_driver_1 = None
        self.fastest_driver_2 = None
        self.telemetry_driver_1 = None
        self.telemetry_driver_2 = None
        self.__delta_time = None
        self.__ref_tel = None
        self.team_driver_1 = None
        self.team_driver_2 = None
        self.team_color_1 = None
        self.team_color_2 = None
        self.__telemetry = None
        self.line_collection = None

    def _load_session(self):
        """Loads the session Data into cache and assigns telemetry attributes"""

        # Enable the cache
        ff1.Cache.enable_cache(self.cache_path)
        self.session = ff1.get_session(self.year, self.grand_prix, "Q")
        self.session.load()

        # Assign Driver laps, fastest lap, and telemetry

        self.laps_driver_1 = self.session.laps.pick_driver(self.driver_1)
        self.laps_driver_2 = self.session.laps.pick_driver(self.driver_2)

        self.fastest_driver_1 = self.laps_driver_1.pick_fastest()
        self.fastest_driver_2 = self.laps_driver_2.pick_fastest()

        self.telemetry_driver_1 = self.fastest_driver_1.get_telemetry()
        self.telemetry_driver_2 = self.fastest_driver_2.get_telemetry()

    def _merge_telemetries(self):
        """Merges the Two Drivers telemetries into a single data frame"""

        self.telemetry_driver_1["Driver"] = self.driver_1
        self.telemetry_driver_2["Driver"] = self.driver_2

        self.__telemetry = pd.concat([self.telemetry_driver_1, self.telemetry_driver_2])

    def _calculate_gap(self):
        """Calculates gap in fastest lap between drivers along the lap"""

        self.__delta_time, self.__ref_tel, _ = delta_time(
            self.fastest_driver_1, self.fastest_driver_2
        )

    def _identify_team_colors(self):
        """Identifies Team Colors For the Chosen Drivers"""

        team_1 = self.laps_driver_1["Team"].iloc[0]
        team_2 = self.laps_driver_2["Team"].iloc[0]
        self.team_color_1 = plotting.team_color(team_1)
        if team_1 != team_2:
            self.team_color_2 = plotting.team_color(self.laps_driver_2["Team"].iloc[0])
        else:
            self.team_color_2 = "#D3D3D3"

    def process_minisectors(
        self,
        num_minisectors: int = 25,
    ):
        """Calculates Minisectors for Plot

        Args:
            num_minisectors (int): Number of sector into whic the track is split.
                Defaults to 25.
        """

        self._merge_telemetries()

        # Calculate minisectors
        total_distance = max(self.__telemetry["Distance"])
        minisector_length = total_distance / num_minisectors

        # Assign a minisector number to every row in the telemetry dataframe
        self.__telemetry["Minisector"] = self.__telemetry["Distance"].apply(
            lambda dist: (int((dist // minisector_length) + 1))
        )

        # Calculate minisector speeds per driver
        average_speed = (
            self.__telemetry.groupby(["Minisector", "Driver"])["Speed"]
            .mean()
            .reset_index()
        )

        # Per minisector, find the fastest driver
        fastest_driver = average_speed.loc[
            average_speed.groupby(["Minisector"])["Speed"].idxmax()
        ]

        fastest_driver = fastest_driver[["Minisector", "Driver"]].rename(
            columns={"Driver": "Fastest_driver"}
        )

        # Merge the fastest_driver dataframe to the telemetry dataframe on minisector
        self.__telemetry = self.__telemetry.merge(fastest_driver, on=["Minisector"])
        self.__telemetry = self.__telemetry.sort_values(by=["Distance"])

        # Plot can only work with integers, we need to convert the driver abbreviations
        # to integers (1 or 2)
        self.__telemetry.loc[
            self.__telemetry["Fastest_driver"] == self.driver_1, "Fastest_driver_int"
        ] = 1
        self.__telemetry.loc[
            self.__telemetry["Fastest_driver"] == self.driver_2, "Fastest_driver_int"
        ] = 2

        # Get the x and y coordinates
        x_values = np.array(self.__telemetry["X"].values)
        y_values = np.array(self.__telemetry["Y"].values)

        # Convert the coordinates to points, and then concat them into segments
        points = np.array([x_values, y_values]).T.reshape((-1, 1, 2))
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        fastest_driver_array = (
            self.__telemetry["Fastest_driver_int"].to_numpy().astype(float)
        )

        # The segments we just created can now be colored according to the fastest
        # driver in a minisector
        cmap = ListedColormap([self.team_color_1, self.team_color_2])
        lc_comp = LineCollection(segments, norm=plt.Normalize(1, cmap.N + 1), cmap=cmap)
        lc_comp.set_array(fastest_driver_array)
        lc_comp.set_linewidth(5)

        self.line_collection = lc_comp

    def process_telemetry(self):
        """Runs the Pipeline for Obtaining and Processing Telemetry"""

        self._load_session()
        self._calculate_gap()
        self._identify_team_colors()

    def compare_telemetry(
        self,
        figsize: List[int] = None,
        height_ratios: List[int] = None,
        path: str = None,
    ):
        """Plots the Telemetry Comparison

        Args:
            figsize (List[int], optional): Size of the plot. Defaults to [20, 15].
            height_ratios (List[int], optional): Relative heights of subplots.
                Defaults to [1, 3, 2, 1, 2, 2, 1].
            path (str, optional):  Directory where plots are to be stored.
            Defaults to 'Results/'
        """

        if figsize is None:
            figsize = [20, 15]

        if height_ratios is None:
            height_ratios = [1, 3, 2, 1, 2, 2, 1]

        if path is None:
            path = (
                os.getcwd()
                + f"/Results/{self.year} {self.session.event.EventName}/Qualifying/"
            )
        else:
            path = (
                path
                + f"/Results/{self.year} {self.session.event.EventName}/Qualifying/"
            )

        plt.rcParams["figure.figsize"] = figsize

        _, axes = plt.subplots(7, gridspec_kw={"height_ratios": height_ratios})

        # Set The Title
        axes[0].title.set_text(
            f"{self.year} {self.session.event.EventName} Qualifying \n"
            f"{self.driver_1} / {self.driver_2} Telemetry Comparison"
        )

        # Subplot 1: The delta
        axes[0].plot(
            self.__ref_tel["Distance"], self.__delta_time, color=self.team_color_2
        )
        axes[0].axhline(0, color=self.team_color_1)
        axes[0].set(ylabel=f"Gap to {self.driver_1} (s)")

        # Subplot 2: Speed
        axes[1].plot(
            self.telemetry_driver_1["Distance"],
            self.telemetry_driver_1["Speed"],
            label=self.driver_1,
            color=self.team_color_1,
        )
        axes[1].plot(
            self.telemetry_driver_2["Distance"],
            self.telemetry_driver_2["Speed"],
            label=self.driver_2,
            color=self.team_color_2,
        )
        axes[1].set(ylabel="Speed (km/h")
        axes[1].legend(loc="lower right")

        # Subplot 3: Throttle
        axes[2].plot(
            self.telemetry_driver_1["Distance"],
            self.telemetry_driver_1["Throttle"],
            label=self.driver_1,
            color=self.team_color_1,
        )
        axes[2].plot(
            self.telemetry_driver_2["Distance"],
            self.telemetry_driver_2["Throttle"],
            label=self.driver_2,
            color=self.team_color_2,
        )
        axes[2].set(ylabel="Throttle (%)")

        # Subplot 4: Brake
        axes[3].plot(
            self.telemetry_driver_1["Distance"],
            self.telemetry_driver_1["Brake"],
            label=self.driver_1,
            color=self.team_color_1,
        )
        axes[3].plot(
            self.telemetry_driver_2["Distance"],
            self.telemetry_driver_2["Brake"],
            label=self.driver_2,
            color=self.team_color_2,
        )
        axes[3].set(ylabel="Brake (On / Off")

        # Subplot 5: Gear
        axes[4].plot(
            self.telemetry_driver_1["Distance"],
            self.telemetry_driver_1["nGear"],
            label=self.driver_1,
            color=self.team_color_1,
        )
        axes[4].plot(
            self.telemetry_driver_2["Distance"],
            self.telemetry_driver_2["nGear"],
            label=self.driver_2,
            color=self.team_color_2,
        )
        axes[4].set(ylabel="Gear")

        # Subplot 6: RPM
        axes[5].plot(
            self.telemetry_driver_1["Distance"],
            self.telemetry_driver_1["RPM"],
            label=self.driver_1,
            color=self.team_color_1,
        )
        axes[5].plot(
            self.telemetry_driver_2["Distance"],
            self.telemetry_driver_2["RPM"],
            label=self.driver_2,
            color=self.team_color_2,
        )
        axes[5].set(ylabel="RPM")

        # Subplot 7: DRS
        axes[6].plot(
            self.telemetry_driver_1["Distance"],
            self.telemetry_driver_1["DRS"],
            label=self.driver_1,
            color=self.team_color_1,
        )
        axes[6].plot(
            self.telemetry_driver_2["Distance"],
            self.telemetry_driver_2["DRS"],
            label=self.driver_2,
            color=self.team_color_2,
        )
        axes[6].set(ylabel="DRS")
        axes[6].set(xlabel="Lap distance (meters)")

        for a in axes.flat:
            a.label_outer()

        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(path + f"Telemetry_Comparison_{self.driver_1}_{self.driver_2}.png")

        plt.show()

    def compare_minisectors(
        self,
        figsize: List[int] = None,
        path: str = None,
    ):
        """Plots Mini-Sectors Comparison

        Args:
            figsize (List[int], optional): Size of the plot. Defaults to [20, 15]
            path (str, optional):  Directory where plots are to be stored.
            Defaults to 'Results/
        """

        if figsize is None:
            figsize = [18, 10]

        if path is None:
            path = (
                os.getcwd()
                + f"/Results/{self.year} {self.session.event.EventName}/Qualifying/"
            )
        else:
            path = (
                path
                + f"/Results/{self.year} {self.session.event.EventName}/Qualifying/"
            )

        # Plot the line collection and style the plot
        plt.gca().add_collection(self.line_collection)
        plt.axis("equal")
        plt.box(False)
        plt.tick_params(labelleft=False, left=False, labelbottom=False, bottom=False)

        # Add a colorbar for as legend
        cbar = plt.colorbar(mappable=self.line_collection, boundaries=np.arange(1, 4))
        cbar.set_ticks([1, 2])
        cbar.set_ticklabels([self.driver_1, self.driver_2])

        if not os.path.exists(path):
            os.makedirs(path)

        plt.savefig(path + f"Minisector_Comparison_{self.driver_1}_{self.driver_2}.png")

        plt.show()
