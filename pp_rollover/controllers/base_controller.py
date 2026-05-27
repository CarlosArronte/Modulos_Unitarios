from abc import ABC, abstractmethod


class BaseController(ABC):
    """
    Base interface for any vehicle controller.
    """

    @abstractmethod
    def plan(self, pose_x, pose_y, pose_theta, **kwargs):
        """
        Compute control action.

        Returns
        -------
        speed : float
        steering : float
        """
        raise NotImplementedError

    @abstractmethod
    def render_waypoints(self, env_renderer):
        """
        Optional visualization hook.
        """
        raise NotImplementedError
