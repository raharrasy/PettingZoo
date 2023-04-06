from pettingzoo.mpe import simple_tag_no_agent_v2

if __name__=="__main__":
    parallel_env = simple_tag_no_agent_v2.parallel_env(max_cycles=100)
    observations = parallel_env.reset()
    counter = 0

    print(parallel_env.agents)
    while parallel_env.agents:
        counter += 1
        actions = {}
        
        #print("Observations: ", observations)
        # Chasing policies
        for agent in observations.keys():
            prey_location_x, prey_location_y = observations[agent][-4], observations[agent][-3]
            longest_distance = max(abs(prey_location_x), abs(prey_location_y))
            if longest_distance == abs(prey_location_x):
                if prey_location_x < 0:
                    actions[agent] = 1
                else:
                    actions[agent] = 2
            else:
                if prey_location_y < 0:
                    actions[agent] = 3
                else:
                    actions[agent] = 4

        #actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
