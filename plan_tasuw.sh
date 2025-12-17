task=point_maze
model_name="2025-10-30/00-47-09"
n_evals=100
planner=cem # gd, cem
goal_H=5
goal_source='random_state'
planner.opt_steps=10

echo "model_name: $model_name, n_evals: $n_evals, planner: $planner, goal_H: $goal_H, goal_source: $goal_source, planner.opt_steps: $planner.opt_steps" >> logs/plan_$task.log

nohup python plan.py model_name="$model_name" n_evals="$n_evals" planner="$planner" goal_H="$goal_H" goal_source="$goal_source" planner.opt_steps="$planner.opt_steps" > logs/plan_$task.log 2>&1 &