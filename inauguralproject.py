#Gammel løsning til opg 3

# setting w_M
wM = model.par.wM

# calculate HF/HM for different values of wF
HFHM = []
log_wfwm = []

for wF in model.par.wF_vec:
    model.par.wF = wF
    opt = model.solve_cont()
    HFHM.append(np.log(opt.HF/opt.HM))
    log_wfwm.append(np.log(wF/wM))

# print solution 
print('Optimal choices:')
print(HFHM)
print(log_wfwm)

# plotting
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(log_wfwm,HFHM)
ax.set_title('Figure 4')
ax.set_xlabel('log rel wages')
ax.set_ylabel('log hours');

# save results into opt values
        opt.LM = res.x[0]
        opt.HM = res.x[1]
        opt.LF = res.x[2]
        opt.HF = res.x[3]

        # print out the values of opt
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')