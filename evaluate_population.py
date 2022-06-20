import easygui
import pickle
from ai_safety_gridworlds.helpers import factory
import yaml
#
from simulate import Model, simulate













if __name__ == '__main__':

    path = easygui.fileopenbox()
    #path = "/home/mathias/real_experiments/rocks_diamonds/generic/ME/run_9/final.p"
    #path = "/home/mathias/real_experiments/tomato_watering/new_ts/ME"

    pop_path = path[0:len(path)-12] + "/run_1/final.p"

    with open(pop_path, 'rb') as f:
        data = pickle.load(f)

    config_filename = path

    config = yaml.safe_load(open(config_filename))
    config["episodes_per_eval"] = 1

    valids = 0
    for ind in data['container']:

        model = Model(config)
        model.set_model_params(ind)
        env = factory.get_environment_obj(config["env_name"])

        scores = simulate(model, env,num_episodes=1, conf=config)
        print(scores["valid"], scores["meanAvgReward"], scores["ss"])
        if scores["valid"]:
            valids += 1

    print(valids)
    #for i in range(N_RUNS):
    #
    #    logger = main()
    #    logs['run_' + str(i)] = logger

    # pickle_out = open(experiment_path + "/logs_" + date + ".pkl", 'wb')
    # pickle.dump(logs, pickle_out)
    # pickle_out.close()


