import numpy as np
from copy as deepcopy

from .refine import box_edges
from .rectangles import to_rec

# Return False only if the polarity is wrong
compute_var_polarity = lambda stleval, bot, top: stleval(bot) <= stleval(top)

# Go through each edge fo teh rectangle. Do not revisit and update with an
# edge's polarity data if that edge has already been set to False
def compute_polarity(stleval, r):
    # stleval: an evaluator of the STL formula
    # r: bounding rectangle
    pol = np.repeat(True, len(r.top))
    edges = list(box_edges(r))
    for edge in edges:
        edge_index = (np.array(edge.top) - np.array(edge.bot)).argmin()
        if pol[edge_index]:
            pol[edge_index] = compute_var_polarity(stleval=stleval,
                                                   bot=edge.bot, top=edge.top)

    return pol

# Project from unit cube
project_from_ub = lambda r, point: (np.array(r.top) - np.array(r.bot))*np.array(
    point) + np.array(r.bot)

# Project to unit cube
project_to_ub = lambda r, theta: (np.array(theta) - np.array(r.bot))/(
    np.array(r.top)- np.array(r.bot))


# Update the rectangle based on polarity
def update_lo_hi(stleval, r):
    var_polarities = compute_polarity(stleval, r)
    updated_bot = r.top*(var_polarities) + r.top*(~var_polarities)*-1
    updated_top = r.bot*(~var_polarities)*-1 + r.top*(var_polarities)

    intervals = [(b,t) for b,t in zip(updated_bot, updated_top)]

    return to_rec(intervals), var_polarities

'''
# Monotone threshold function
def thres_func(stleval, r):
    r_updated, var_pol= thres_func(stleval, r)
    def g(stleval, theta):
        updated_theta = theta*(var_pol) + theta*(~var_pol)*-1
        return stleval(updated_theta)
    return g
'''