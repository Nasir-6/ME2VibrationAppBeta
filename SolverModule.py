import numpy as np
import matplotlib.pyplot as plt


def checker(k=10, m=1, dampRatio=0):
    if m > 0 and k >= 0 and dampRatio >= 0:
        solvable = True
        print("Solvable")
    else:
        solvable = False
        print("No Solution")
    return solvable


def SDOFsolver(k=10, m=1, dampRatio=0.1, x0=0.1, tend=5):
    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    tlim = 30
    if tend < tlim:   #30 is limit (Change this once I have a value)
        t = np.linspace(0, tend, 10000)
    else:
        t = np.linspace(0, tlim, 10000)
    x = t.copy()

    solvable = checker (k, m, dampRatio)
    if solvable:
        if dampRatio == 0:
            x = x0 * np.cos(wn * t)
            solutionType = "Undamped Solution"
        elif 1 > dampRatio > 0:
            solutionType = "Under Damped Solution"
            wd = wn * np.sqrt(1 - dampRatio ** 2)
            A = x0
            B = dampRatio * A / wd
            x = np.exp(-dampRatio * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t))
        elif dampRatio == 1:
            solutionType = "Critically Damped Solution"
            A = x0
            B = A*wn
            x = (A + B*t)*np.exp(-wn*t)
        elif dampRatio > 1:
            solutionType = "Over Damped Solution"
            A = x0 * (dampRatio + np.sqrt(dampRatio**2 - 1))/(2*np.sqrt(dampRatio**2 -1))
            B = x0 - A
            x = A*np.exp((-dampRatio + np.sqrt(dampRatio**2 - 1))*wn*t) + B*np.exp((-dampRatio - np.sqrt(dampRatio**2 - 1))*wn*t)
        else:
            solutionType = "Unaccounted for Solution"
    else:
        solutionType = "Not solvable"
    return x, t, solvable, solutionType


def SDOFplotter(t, x, solvable=False):
    if solvable:
        fig, ax = plt.subplots()
        plt.plot(t, x)
        plt.xlabel("Time (s^-1)")
        plt.ylabel("Displacement (m)")

    else:
        print("WTH IS THIS!")


def forcedSolver(m=10, k=10 ** 6, c=100, x0=0, Famp=10, wHz=5, tend=1):
    tlim = 30
    if tend < tlim:  # 30 is limit (Change this once I have a value)
        t = np.linspace(0, tend, 10000)
    else:
        t = np.linspace(0, tlim, 10000)
    x = t.copy()

    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    dampRatio = c / (2 * np.sqrt(k * m))
    wd = wn * np.sqrt(1 - dampRatio ** 2)  # Damped frequency
    w = 2 * np.pi * wHz  # Conv Forced freq from Hz into rad/s

    solvable = checker(k, m, dampRatio)
    if solvable:
        # Solving for Complete Forced Solution
        # Displacement amplitdue from force ONLY
        x0f = Famp / np.sqrt((k - m * w ** 2) ** 2 + (c * w) ** 2)
        phasef = np.arctan(c * w / (k - m * w ** 2))

        A = x0 - x0f * np.sin(-phasef)
        B = (dampRatio * wn * A - x0f * w * np.cos(-phasef)) / wd

        x = np.exp(-dampRatio * wn * t) * (A * np.cos(wd * t) + B * np.sin(wd * t)) + x0f * np.sin(w * t - phasef)

        # Only the Forcing amplitude and it's relevant displacment
        # Shorter time scale, tf so can see phase shift
        tf = np.linspace(0, 3, 1000)
        F = Famp * np.sin(w * tf)
        xf = x0f * np.sin(w * tf - phasef)

        solutionType = "Forced Response"
    else:
        solutionType = "Not solvable"
    return x, t, F, xf, tf, solvable, solutionType

def forcedPlot(t=0, x=0, F=0, xf=0, tf=0,  solvable=False, plotCombinedResponse=False, wHz=0):
    if solvable:
        if plotCombinedResponse:
            fig, ax = plt.subplots()
            plt.plot(t, x)
            plt.xlabel("Time (s^-1)")
            plt.ylabel("Displacement (m)")
            title = "Forced Time Response for w = {} Hz ".format(wHz)
            plt.title(title)
        else:
            # Now plotting force amp and displacement from Force together
            fig, ax = plt.subplots()
            plt.plot(tf, xf)
            plt.plot(tf, F)
            plt.xlabel("Time (s^-1)")
            plt.ylabel("Displacement (m)/Force (N)")

    else:
        print("WTH IS THIS!")
    return


# FRF ===========================

def FRFSolver(m=10, k=10, dampRatios=np.array([0.25,0.15,0.5]), wantNormalised = False):

    solvable = True
    for dampRat in dampRatios:
        solvable = checker(k=10, m=1, dampRatio=dampRat) and solvable # If any fails it becomes unsolvable

    wn = np.sqrt(k / m)  # Natural Freq of spring mass system
    w = np.linspace(0, 10, 10000)
    r = w / wn

    if solvable:
        amp = np.zeros((len(dampRatios), len(w)))
        phase = np.zeros((len(dampRatios), len(w)))
        if wantNormalised:
            row = 0
            for dampRat in dampRatios:
                print(dampRat)
                amp[row, :] = 1 / np.sqrt((1 - r ** 2) ** 2 + (2 * dampRat * r) ** 2)
                phase[row, :] = np.arctan(-2 * dampRat * r / (1 - r ** 2))
                phase[phase > 0] = phase[phase > 0] - np.pi
                row = row + 1
        else:
            row = 0
            for dampRat in dampRatios:
                c = dampRat * 2 * np.sqrt(k*m)
                print(dampRat)
                amp[row, :] = 1 / np.sqrt((k - m*w**2) ** 2 + (c*w) ** 2)
                phase[row, :] = np.arctan(-c*w / (k - m*w**2))
                phase[phase > 0] = phase[phase > 0] - np.pi
                row = row + 1

    else:
        print("WTH IS THIS!")
    return amp, phase, r, w, solvable

def FRFPlot(solvable, amp, phase, dampRatios=np.array([0.25,0.15,0.5]), r=np.linspace(0,10,10000), w=np.linspace(0,10,10000), wantNormalised=False):

    if solvable:
        f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        if wantNormalised:
            row = 0
            for dampRat in dampRatios:
                # Plotting
                linename = "Damping Ratio = {}".format(dampRat)
                ax1.plot(r, amp[row, :], label=linename)
                ax2.plot(r, phase[row, :], label=linename)
                row = row + 1
            plt.xlabel("r = w/wn")
            ax1.set(ylabel='kx/F')
            ax2.set(ylabel='Phase (rad)')
            ax1.legend()
            ax2.legend()

        else:
            row = 0
            for dampRat in dampRatios:
                # Plotting
                linename = "Damping Ratio = {}".format(dampRat)
                ax1.plot(w, amp[row, :], label=linename)
                ax2.plot(w, phase[row, :], label=linename)
                row = row + 1
            plt.xlabel("w (rad/s)")
            ax1.set(ylabel='x/F (m/N)')
            ax2.set(ylabel='Phase (rad)')
            ax1.legend()
            ax2.legend()

    else:
        print("WTH IS THIS!")
    return


# Transmissibility For diff damping ratios ========================

def TransmissibilitySolver(dampRatios = np.array([0.25,0.15,0.5])):
    solvable = True
    for dampRat in dampRatios:
        solvable = checker(k=10, m=1, dampRatio=dampRat) and solvable  # If any fails it becomes unsolvable

    if solvable:
        r = np.linspace(0, 5, 1000)
        T = np.zeros((len(dampRatios), len(r)))
        row = 0
        for dampRat in dampRatios:
            print(dampRat)
            T[row, :] = np.sqrt((1 + (2 * dampRat * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * dampRat * r) ** 2))
            row = row + 1
    else:
        print("WTH IS THIS!")

    return r, T, solvable

def TransmissibilityPlot(dampRatios, r,T,solvable):

    if solvable:
        row = 0
        fig, ax = plt.subplots()
        for dampRat in dampRatios:
            linename = "Damping Ratio = {}".format(dampRat)
            plt.plot(r, T[row, :], label=linename)
            row = row + 1


        plt.xlabel("Excitation ratio")
        plt.ylabel("Transmissibility")
        ax.legend()
    else:
        print("WTH IS THIS!")


# Transmissibility Force and input Force Time history plot for one Frewuency ============

def TransmissibilityTimeHistorySolver(k=10, m=1, dampRat=0.1, Famp=0.1, w=5, tend=1):
    solvable = checker(k=10, m=1, dampRatio=dampRat)
    if solvable:
        tlim = 30
        if tend < tlim:  # 30 is limit (Change this once I have a value)
            t = np.linspace(0, tend, 10000)
        else:
            t = np.linspace(0, tlim, 10000)

        wn = np.sqrt(k / m)  # Natural Freq of spring mass system
        r = w/wn
        c = dampRat * 2 * np.sqrt(k * m)
        T = np.sqrt((1 + (2 * dampRat * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * dampRat * r) ** 2))
        theta = np.arctan(abs(c*w/(k - m*w**2))) - np.arctan(c*w/k)
        phaseDeg = theta*180/np.pi
        print("Phase Shift = {} Degrees".format(phaseDeg))
        print(np.arctan(c*w/(k - m*w**2)))
        print(np.arctan(c*w/k)*180/np.pi)
        F = Famp * np.sin(w*t)
        Ft = T* Famp * np.sin(w*t - theta)
    else:
        print("WTH IS THIS!")

    return F, Ft, t, r, solvable


def TransmissibilityTimeHistoryPlot(F, Ft, t, r, dampRat, solvable):

    if solvable:
        fig, ax = plt.subplots()
        plt.plot(t, F, label="Input Force")
        plt.plot(t, Ft, label="Transmitted Force")
        plt.xlabel("Time (s)")
        plt.ylabel("Force Amplitude (N)")
        ax.legend()
        title = "Transmissibility Force Time history at r={} and dampRat = {}".format(r,dampRat)
        plt.title(title)
    else:
        print("WTH IS THIS!")




# Base Excitation

def BaseExciteSolver(m,k,dampRatios= np.array([0.25,0.15,0.5]), normalised = False):
    solvable = True
    for dampRat in dampRatios:
        solvable = checker(k=k, m=m, dampRatio=dampRat) and solvable  # If any fails it becomes unsolvable

    if solvable:
        w = np.linspace(0, 5, 1000)
        wn = np.sqrt(k / m)  # Natural Freq of spring mass system
        r = w/wn
        if normalised:
            r = np.linspace(0, 5, 1000)
            MT = np.zeros((len(dampRatios), len(r)))
            row = 0
            for dampRat in dampRatios:
                print(dampRat)
                MT[row, :] = np.sqrt((1 + (2 * dampRat * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * dampRat * r) ** 2))
                row = row + 1
        else:
            w = np.linspace(0, 5, 1000)
            MT = np.zeros((len(dampRatios), len(w)))
            row = 0
            for dampRat in dampRatios:
                print(dampRat)
                c = dampRat * 2 * np.sqrt(k*m)
                MT[row, :] = np.sqrt((k**2 + (c*w) ** 2) / ((k - m*w**2) ** 2 + (c*w) ** 2))
                row = row + 1
    else:
        print("WTH IS THIS!")

    return w, r, MT, solvable

def BaseExcitePlot(dampRatios,w, r, MT, solvable, normalised):

    if solvable:
        if normalised:
            row = 0
            fig, ax = plt.subplots()
            for dampRat in dampRatios:
                linename = "Damping Ratio = {}".format(dampRat)
                plt.plot(r, MT[row, :], label=linename)
                row = row + 1
            plt.xlabel("r = w/wn")
            plt.ylabel("Motion Transmissibility x0/y0")
            ax.legend()
        else:
            row = 0
            fig, ax = plt.subplots()
            for dampRat in dampRatios:
                linename = "Damping Ratio = {}".format(dampRat)
                plt.plot(w, MT[row, :], label=linename)
                row = row + 1

            plt.xlabel("w (rad/s)")
            plt.ylabel("Motion Transmissibility x0/y0")
            ax.legend()

    else:
        print("WTH IS THIS!")




# Base Excitation Time history of y and x

def BaseExciteTimeHistSolver(m,k,dampRatio, y0, w, tend):
    solvable = True

    solvable = checker(k=k, m=m, dampRatio=dampRatio) and solvable  # If any fails it becomes unsolvable

    if solvable:
        wn = np.sqrt(k / m)  # Natural Freq of spring mass system
        r = w/wn
        c = dampRatio * 2 * np.sqrt(k * m)
        tlim = 30
        if tend < tlim:  # 30 is limit (Change this once I have a value)
            t = np.linspace(0, tend, 10000)
        else:
            t = np.linspace(0, tlim, 10000)

        y = y0*np.sin(w*t)
        x0 = y0 * np.sqrt((1 + (2 * dampRatio * r) ** 2) / ((1 - r ** 2) ** 2 + (2 * dampRatio * r) ** 2))
        phase = np.arctan(c*w/(k-m*w**2)) - np.arctan(c*w/k)
        x = x0*np.sin(w*t - phase)
        print("Phase Shift = {}".format(phase))

    else:
        print("WTH IS THIS!")

    return y, x, t, r, solvable


def BaseExciteTimeHistPlot(x, y, t, r, dampRatio, solvable):

    if solvable:

        fig, ax = plt.subplots()
        plt.plot(t, y, label="y (base)")
        plt.plot(t, x, label="x (mass)")

        plt.xlabel("Time (s)")
        plt.ylabel("Displacement (m)")
        title = "Base Excitation Time history at r={} and dampRat = {}".format(r, dampRatio)
        plt.title(title)
        ax.legend()
    else:
        print("WTH IS THIS!")