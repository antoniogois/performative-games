import math


class CRD:
    """An instance of the Collective Risk Dilemma"""
    def __init__(self,
                 e: float,
                 r: float,
                 c: float,
                 t: float):
        """
        :param e: float, initial endowment
        :param r: float in [0,1], risk of everyone losing everything
        :param c: float, cost of cooperating
        :param t: float in [0,1], fraction of cooperators in group required
        """
        self.e = e
        self.r = r
        self.c = c
        self.t = t

    def get_threshold(self, group_size: int):
        """
        compute minimum number of cooperators required, given group size and fraction t of cooperators required
        :param group_size:
        :return:
        """
        return math.ceil(self.t * group_size)

    def defector_payoff(self,
                        cooperators: int,
                        group_size: int):
        """

        :param cooperators: total number of cooperators in group
        :param group_size:
        :return: float, payoff of defector
        """
        assert cooperators <= group_size
        threshold = self.get_threshold(group_size)
        success = cooperators >= threshold
        if success:
            return self.e
        else:
            return self.e * (1 - self.r)

    def cooperator_payoff(self,
                          cooperators: int,
                          group_size: int):
        """

        :param cooperators: total number of cooperators in group, including self
        :param group_size:
        :return: float, payoff of cooperator
        """
        return self.defector_payoff(cooperators, group_size) - self.c * self.e
