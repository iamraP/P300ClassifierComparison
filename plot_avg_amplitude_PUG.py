import matplotlib.pyplot as plt

col_list=["#377eb8","#e41a1c"]
fig, Cz = plt.subplots(1, 1, figsize=(6, 5), sharey=True)
for e in ["Cz"]:
    electrode = picks.index(e)
    eval(e).plot(times, t_avg[electrode], label="Target (n="+ str(t.shape[0])+")", c=col_list[1])
    eval(e).plot(times, nt_avg[electrode], label="Non-Target (n="+ str(nt.shape[0])+")", c=col_list[0])
    t_std = np.std(t[:, electrode, :], axis=0)
    nt_std = np.std(nt[:, electrode, :], axis=0)
    t_ci = 1.96 * t_std / np.sqrt(t.shape[0])
    nt_ci = 1.96 * nt_std / np.sqrt(nt.shape[0])
    eval(e).fill_between(times, t_avg[electrode] -t_ci, t_avg[electrode] + t_ci, alpha=0.3, color=col_list[1])
    eval(e).fill_between(times, nt_avg[electrode] - nt_ci, nt_avg[electrode] + nt_ci, alpha=0.3, color=col_list[0])
    eval(e).axvline(x=0, c='k', lw=0.5)
    eval(e).axhline(y=0, c='k', lw=0.5)
    eval(e).axvspan(0.35, 0.6, facecolor="#808080", alpha=0.3)
    eval(e).set_xlabel("seconds")
    eval(e).set_ylabel("µV")
    eval(e).margins(x=0)
    eval(e).set_title(e)
Fz.set_ylabel("µV")

Cz.legend(title="Average over all sessions", loc='center', bbox_to_anchor=(0.5, 1.16), ncol=2)
# fig.suptitle("Average over all sessions")
fig.tight_layout()
#plt.show()
plt.savefig(r"C:\Users\map92fg\Documents\Software\P300_Classification\data_thesis\PUG_average_Cz.svg",format="svg")