import utils

exploration_rate_type_to_obj = {'constant': float,
                                'decay': utils.ExplorationRateDecay,
                                'dabney': float}
