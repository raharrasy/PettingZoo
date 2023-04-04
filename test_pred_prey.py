from pettingzoo.mpe import simple_tag_no_agent_v2

if __name__=="__main__":
    parallel_env = simple_tag_no_agent_v2.parallel_env()
    observations = parallel_env.reset()
    counter = 0

    print(parallel_env.agents)
    while parallel_env.agents:
        counter += 1
        print("Counter :", counter)
        actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents}  # this is where you would insert your policy
        observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
