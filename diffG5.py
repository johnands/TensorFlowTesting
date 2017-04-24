from sympy.utilities.codegen import codegen
from sympy import *

# variables
xj, yj, zj, xk, yk, zk =  symbols('xj yj zj xk yk zk')

# parameters
eta, Rc, zeta, Lambda = symbols('eta Rc zeta Lambda')

# substituion parameters
Rj, Rk, Rj2, Rk2, RjDotRk, CosTheta, Fcj, Fck, dFcj, dFck = \
    symbols('Rj Rk Rj2 Rk2 RjDotRk CosTheta Fcj Fck dFcj dFck')

rj = sqrt(xj**2 + yj**2 + zj**2)
rk = sqrt(xk**2 + yk**2 + zk**2)

fcj = 0.5*cos(pi*rj/Rc) + 0.5
fck = 0.5*cos(pi*rk/Rc) + 0.5

rjDotrk = (xj*xk + yj*yk + zj*zk)
cosTheta = rjDotrk / (rj*rk)

F1 = 2**(1-zeta) * (1 + Lambda*cosTheta)**zeta
F2 = exp(-eta*(rj**2 + rk**2))
F3 = fcj*fck

G5 = F1*F2*F3

dG5dxj = diff(G5, xj)
dG5dxj = dG5dxj.subs(xj**2 + yj**2 + zj**2, Rj2)
dG5dxj = dG5dxj.subs(xk**2 + yk**2 + zk**2, Rk2)
dG5dxj = dG5dxj.subs(sqrt(Rj2), Rj)
dG5dxj = dG5dxj.subs(sqrt(Rk2), Rk)
dG5dxj = dG5dxj.subs(xj*xk + yj*yk + zj*zk, RjDotRk)
dG5dxj = dG5dxj.subs(RjDotRk/(Rj*Rk), CosTheta)
dG5dxj = dG5dxj.subs(0.5*cos(pi*Rj/Rc) + 0.5, Fcj)
dG5dxj = dG5dxj.subs(0.5*cos(pi*Rk/Rc) + 0.5, Fck)
dG5dxj = dG5dxj.subs(sin(pi*Rj/Rc)/(Rc), -2*dFcj/pi)
print simplify(dG5dxj)

"""dG4dyj = diff(G4, yj)
dG4dyj = dG4dyj.subs(xj**2 + yj**2 + zj**2, Rj2)
dG4dyj = dG4dyj.subs(xk**2 + yk**2 + zk**2, Rk2)
dG4dyj = dG4dyj.subs((xj-xk)**2 + (yj-yk)**2 + (zj-zk)**2, Rjk2)
dG4dyj = dG4dyj.subs(sqrt(Rj2), Rj)
dG4dyj = dG4dyj.subs(sqrt(Rk2), Rk)
dG4dyj = dG4dyj.subs(sqrt(Rjk2), Rjk)
dG4dyj = dG4dyj.subs(xj*xk + yj*yk + zj*zk, RjDotRk)
dG4dyj = dG4dyj.subs(RjDotRk/(Rj*Rk), CosTheta)
dG4dyj = dG4dyj.subs(0.5*cos(pi*Rj/Rc) + 0.5, Fcj)
dG4dyj = dG4dyj.subs(0.5*cos(pi*Rk/Rc) + 0.5, Fck)
dG4dyj = dG4dyj.subs(0.5*cos(pi*Rjk/Rc) + 0.5, Fcjk)
dG4dyj = dG4dyj.subs(sin(pi*Rj/Rc)/(Rc), -2*dFcj/pi)
dG4dyj = dG4dyj.subs(sin(pi*Rjk/Rc)/(Rc), -2*dFcjk/pi)
#print simplify(dG4dyj)"""

dG5dxk = diff(G5, xk)
dG5dxk = dG5dxk.subs(xj**2 + yj**2 + zj**2, Rj2)
dG5dxk = dG5dxk.subs(xk**2 + yk**2 + zk**2, Rk2)
dG5dxk = dG5dxk.subs(sqrt(Rj2), Rj)
dG5dxk = dG5dxk.subs(sqrt(Rk2), Rk)
dG5dxk = dG5dxk.subs(xj*xk + yj*yk + zj*zk, RjDotRk)
dG5dxk = dG5dxk.subs(RjDotRk/(Rj*Rk), CosTheta)
dG5dxk = dG5dxk.subs(0.5*cos(pi*Rj/Rc) + 0.5, Fcj)
dG5dxk = dG5dxk.subs(0.5*cos(pi*Rk/Rc) + 0.5, Fck)
dG5dxk = dG5dxk.subs(sin(pi*Rk/Rc)/(Rc), -2*dFck/pi)
print simplify(dG5dxk)

print codegen(("dG5dxj", simplify(dG5dxj)), "C", "file")[0][1]
print codegen(("dG5dxk", simplify(dG5dxk)), "C", "file")[0][1]