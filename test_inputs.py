from vis_nav_game import Action

# IDLE pause between each action
# Semi-decent run with STEP_SIZE = 3, without abs() code
test_inputs = [
    {
        'steps':5,
        'actions':Action.IDLE
    },
    {
        'steps':10,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':10,
        'actions':Action.BACKWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.RIGHT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':41,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.LEFT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':12,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.RIGHT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':12,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.RIGHT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':8,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':8,
        'actions':Action.BACKWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.RIGHT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':40,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.RIGHT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':14,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.LEFT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':15,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.RIGHT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':40,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':27,
        'actions':Action.BACKWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':37,
        'actions':Action.RIGHT
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':12,
        'actions':Action.FORWARD
    },
    {
        'steps':8,
        'actions':Action.IDLE
    },
    {
        'steps':1,
        'actions':Action.QUIT
    },
    {
        'steps':10,
        'actions':Action.IDLE
    },
    {
        'steps':1,
        'actions':Action.QUIT
    },
]

# Test run with no idle between actions
# test_inputs = [
#     {
#         'steps':5,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':10,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':10,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':41,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.LEFT
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':40,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':14,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.LEFT
#     },
#     {
#         'steps':15,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':50,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':1,
#         'actions':Action.QUIT
#     },
#     {
#         'steps':10,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':1,
#         'actions':Action.QUIT
#     },
# ]


# This test run with STEP_SIZE 3 and 4 causes the decompoeEssentialMat error
# test_inputs = [
#     {
#         'steps':5,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':10,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':10,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':41,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.LEFT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':8,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':8,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':40,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':14,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.LEFT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':15,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':50,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':1,
#         'actions':Action.QUIT
#     },
#     {
#         'steps':10,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':1,
#         'actions':Action.QUIT
#     },
# ]



# No IDLE between actions
# test_inputs = [
#     {
#         'steps':5,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':10,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':10,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':41,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.LEFT
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':8,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':8,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':40,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':14,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.LEFT
#     },
#     {
#         'steps':15,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':40,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':27,
#         'actions':Action.BACKWARD
#     },
#     {
#         'steps':37,
#         'actions':Action.RIGHT
#     },
#     {
#         'steps':12,
#         'actions':Action.FORWARD
#     },
#     {
#         'steps':1,
#         'actions':Action.QUIT
#     },
#     {
#         'steps':10,
#         'actions':Action.IDLE
#     },
#     {
#         'steps':1,
#         'actions':Action.QUIT
#     },
# ]