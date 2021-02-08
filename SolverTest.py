from SolverModule import *
import matplotlib.pyplot as plt

# SDOF Damped/Undamped ===============================
# k=10
# m = 5
# dampRatio = 0.3
# x0 = 0.1
# tend = 20
# x, t, solvable, solutionType = SDOFsolver(k, m, dampRatio,  x0, tend)
# SDOFplotter(t, x, solvable)

# FORCED RESPONSE ====================================================
# Michal's Parameters
# m=10
# k=10 ** 6
# c=100
# x0=0
# Famp=10
# wHz=5
# tend=1

m=1
k=50
c= 0.02 * 2* np.sqrt(k*m)   # for damping ratio = 0.02
x0=0.6
Famp=0.2
wHz= np.sqrt(k / m)/(2*np.pi)   # so get resonance
tend=30

# x, t, F, xf, tf, solvable, solutionType = forcedSolver(m, k, c, x0, Famp, wHz, tend)
# To plot the Total displacement against time graph
# plotCombinedResponse = True
# forcedPlot(t, x, F, xf, solvable, plotCombinedResponse, wHz)
# Now only plotting the Force and forced displacemnt ONLY against time
# plotCombinedResponse = False
# forcedPlot(t, x, F, xf, tf, solvable, plotCombinedResponse, wHz)


# FRF PLOT ========================================================
# k = 10  # N/m
# m = 1   #kg
# dampRatios = np.array([0.25,0.15,0.05])
# wantNormalised = False
# amp, phase, r, w, solvable = FRFSolver(m, k, dampRatios, wantNormalised)
# FRFPlot(solvable, amp, phase, dampRatios, r, w, wantNormalised)

# Transmissibility PLot ======================
# dampRatios = np.array([0.25,0.15,0.5])
# r, T, solvable = TransmissibilitySolver(dampRatios = np.array([0.25,0.15,0.5]))
# TransmissibilityPlot(dampRatios, r,T,solvable)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!! NEEDS FIXING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Transmissibikity time plot at single frequnecy =====
# k = 10
# m = 1
# dampRat = 0.3
#
# Famp = 10
# wn = np.sqrt(k / m)  # Natural Freq of spring mass system
# w = wn# * np.sqrt(2) + 1
# tend = 10
# F, Ft, t, r, solvable = TransmissibilityTimeHistorySolver(k,m,dampRat,Famp,w,tend)
# TransmissibilityTimeHistoryPlot(F, Ft, t, r, dampRat, solvable)


# Base ExcitationPLot ======================
# m = 1
# k = 10
# print(np.sqrt(k / m))
# dampRatios = np.array([0.25,0.15,0.5])
# normalised = True
# w, r, MT, solvable = BaseExciteSolver(m,k,dampRatios,normalised)
# BaseExcitePlot(dampRatios, w, r,MT,solvable, normalised)



# Base Excitation Time History x,y plot ====================================
m = 1
k = 10
wn = np.sqrt(k / m)
print(np.sqrt(k / m))
dampRatio = 0.25
y0 = 0.1
# w = 3
w = wn * np.sqrt(2) + 1
tend = 10

y, x, t, r, solvable = BaseExciteTimeHistSolver(m,k,dampRatio, y0, w, tend)
BaseExciteTimeHistPlot(x, y, t, r, dampRatio, solvable)





plt.show()
