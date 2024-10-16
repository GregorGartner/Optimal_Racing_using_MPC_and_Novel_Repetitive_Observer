import casadi as ca

# Create an empty NLP problem to check if IPOPT is available
try:
    solver = ca.nlpsol('solver', 'ipopt', {})  # Empty problem
    print("IPOPT is available.")
except RuntimeError as e:
    print(f"IPOPT is not available: {e}")