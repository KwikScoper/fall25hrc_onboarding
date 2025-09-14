import time
from pathlib import Path
import mujoco as mj
from mujoco import viewer
import numpy as np

def lin_interp(x, values):
    num_values = len(values)

    x = max(0, min(x, num_values - 1))

    i = int(x)
    x_frac = x - i

    #Edge cases for t outside list range
    if i == num_values - 1:
        return values[i]
    
    v1 = values[i]
    v2 = values[i + 1]

    return v1 + (v2 - v1) * x_frac

#---------------------------------------

def view_floor():
    model = mj.MjModel.from_xml_path("xml/floor.xml")
    data = mj.MjData(model)

    dt = model.opt.timestep
    start = time.time()
    t = 0.0

    #PD loop initialization values
    Kp = 5.0
    Kd = 0.1

    gait_period = 4.0

    mid_positions = [0.0, -0.3, -0.6, -0.3, 0.6, 0.3]
    bot_positions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.2]

    #joint trajectory
    freq = 1.0
    amp_mid = 0.5
    amp_bot = 0.8
    roll_angle = 0.3

    #number of joints
    njoints = model.nu


    viewer2 = viewer.launch_passive(model, data)


    while viewer2.is_running():
        # indexing leg joints
        leg_joints = {
            "FL": {"roll": 0, "mid": 4, "bot": 8},
            "BL": {"roll": 1, "mid": 5, "bot": 9},
            "FR": {"roll": 2, "mid": 6, "bot": 10},
            "BR": {"roll": 3, "mid": 7, "bot": 11},
        }

        roll_sign = {"FL": +1, "BL": +1, "FR": -1, "BR": -1}

        phases = {
            "FL": 0,
            "BL": 0.5,
            "FR": 0.5,
            "BR": 0,
        }

        qpos_des = np.zeros(njoints)

        for leg, idxs in leg_joints.items():
            phase = phases[leg]

            cycle_pos = (t / gait_period + phase) % 1.0

            interp_index = cycle_pos * (len(mid_positions) - 1)


            # roll joint
            qpos_des[idxs["roll"]] = roll_angle * roll_sign[leg]
            qpos_des[idxs["roll"]] = roll_angle * roll_sign[leg]

            # mid pitch joint
            mid_interp_val = lin_interp(interp_index, mid_positions)
            qpos_des[idxs["mid"]] = mid_interp_val
            #qpos_des[idxs["mid"]] = amp_mid * np.sin(3 * np.pi * freq * t + phase)
            
            #bot pitch joint
            bot_interp_val = lin_interp(interp_index, bot_positions)
            qpos_des[idxs["bot"]] = bot_interp_val
            #qpos_des[idxs["bot"]] = -0.5 + amp_bot * np.sin(3 * np.pi * freq * t + phase + np.pi/2)



        qx = data.qpos[-njoints:]
        qv = data.qvel[-njoints:]
        qv_des = np.zeros_like(qpos_des)

        tau = Kp * (qpos_des - qx) + Kd * (qv_des - qv)
        data.ctrl[:] = tau


        mj.mj_step(model, data)
        #print(data.qpos)
        t += dt
        time.sleep(dt)
        viewer2.sync()



#        print(model.joint('jtopFL_roll'))
#        print(model.joint('jtopBL_roll'))
#        print(model.joint('jtopFR_roll'))
#        print(model.joint('jtopBR_roll'))
#        print(model.joint('jmidFL_pitch'))
#        print(model.joint('jmidBL_pitch'))
#        print(model.joint('jmidFR_pitch'))
#        print(model.joint('jmidBR_pitch'))
#        print(model.joint('jbotBL_pitch'))
#        print(model.joint('jbotFL_pitch'))
#        print(model.joint('jbotFR_pitch'))
#        print(model.joint('jbotBR_pitch'))
        print("______________")


if __name__ == "__main__":
    view_floor()