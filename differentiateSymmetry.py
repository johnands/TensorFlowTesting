from sympy.utilities.codegen import codegen
from sympy import *

### alternative 1 ###

#xij, yij, zij, xik, yik, zik, xjk, yjk, zjk = symbols('xij yij zij xik yik zik xjk yjk zjk')
#eta, Rc, zeta, Lambda = symbols('eta Rc zeta Lambda')
#rij, rik, rjk, rij2, rik2, rjk2, rijDotRik, cosTheta = symbols('rij, rik, rjk, rij2, rik2, rjk2, rijDotRik, cosTheta')

"""G4 = 2**(1-zeta) * (1 + Lambda*((xij*xik + yij*yik + zij*zik)/(sqrt(xij**2 + yij**2 + zij**2)*sqrt(xik**2 + yik**2 + zik**2))))**(zeta) * exp(-eta*(xij**2 + yij**2 + zij**2 + xik**2 + yik**2 + zik**2 + xjk**2 + yjk**2 + zjk**2)) * 0.5*(cos((pi*sqrt(xij**2 + yij**2 + zij**2))/Rc) + 1) * 0.5*(cos((pi*sqrt(xik**2 + yik**2 + zik**2))/Rc) + 1) + 0.5*(cos((pi*sqrt(xjk**2 + yjk**2 + zjk**2))/Rc) + 1)"""

"""dG4dxij = diff(G4, xij)
dG4dxij = dG4dxij.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
dG4dxij = dG4dxij.subs(sqrt(xik**2 + yik**2 + zik**2), rik)
dG4dxij = dG4dxij.subs(sqrt(xjk**2 + yjk**2 + zjk**2), rjk)
dG4dxij = dG4dxij.subs(xij**2 + yij**2 + zij**2, rij2)
dG4dxij = dG4dxij.subs(xik**2 + yik**2 + zik**2, rik2)
dG4dxij = dG4dxij.subs(xjk**2 + yjk**2 + zjk**2, rjk2)
dG4dxij = dG4dxij.subs(xij*xik + yij*yik + zij*zik, rijDotRik)
dG4dxij = dG4dxij.subs(rijDotRik/(rij*rik), cosTheta)
print dG4dxij"""


### alternative 2 ###

# variables
xj, yj, zj, xk, yk, zk =  symbols('xj yj zj xk yk zk')

# parameters
eta, Rc, zeta, Lambda = symbols('eta Rc zeta Lambda')

# substituion parameters
Rj, Rk, Rjk, Rj2, Rk2, Rjk2, RjDotRk, CosTheta, Fcj, Fck, Fcjk, dFcj, dFck, dFcjk = symbols('Rj Rk Rjk Rj2 Rk2 Rjk2 RjDotRk CosTheta Fcj Fck Fcjk dFcj dFck dFcjk')

rj = sqrt(xj**2 + yj**2 + zj**2)
rk = sqrt(xk**2 + yk**2 + zk**2)
rjk = sqrt((xj-xk)**2 + (yj-yk)**2 + (zj-zk)**2)

fcj = 0.5*cos(pi*rj/Rc) + 0.5
fck = 0.5*cos(pi*rk/Rc) + 0.5
fcjk = 0.5*cos(pi*rjk/Rc) + 0.5

rjDotrk = (xj*xk + yj*yk + zj*zk)
cosTheta = rjDotrk / (rj*rk)

F1 = 2**(1-zeta) * (1 + Lambda*cosTheta)**zeta
F2 = exp(-eta*(rj**2 + rk**2 + rjk**2))
F3 = fcj*fck*fcjk

G4 = F1*F2*F3

dG4dxj = diff(G4, xj)
dG4dxj = dG4dxj.subs(xj**2 + yj**2 + zj**2, Rj2)
dG4dxj = dG4dxj.subs(xk**2 + yk**2 + zk**2, Rk2)
dG4dxj = dG4dxj.subs((xj-xk)**2 + (yj-yk)**2 + (zj-zk)**2, Rjk2)
dG4dxj = dG4dxj.subs(sqrt(Rj2), Rj)
dG4dxj = dG4dxj.subs(sqrt(Rk2), Rk)
dG4dxj = dG4dxj.subs(sqrt(Rjk2), Rjk)
dG4dxj = dG4dxj.subs(xj*xk + yj*yk + zj*zk, RjDotRk)
dG4dxj = dG4dxj.subs(RjDotRk/(Rj*Rk), CosTheta)
dG4dxj = dG4dxj.subs(0.5*cos(pi*Rj/Rc) + 0.5, Fcj)
dG4dxj = dG4dxj.subs(0.5*cos(pi*Rk/Rc) + 0.5, Fck)
dG4dxj = dG4dxj.subs(0.5*cos(pi*Rjk/Rc) + 0.5, Fcjk)
dG4dxj = dG4dxj.subs(sin(pi*Rj/Rc)/(Rc), -2*dFcj/pi)
dG4dxj = dG4dxj.subs(sin(pi*Rjk/Rc)/(Rc), -2*dFcjk/pi)
#print simplify(dG4dxj)

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

dG4dxk = diff(G4, xk)
dG4dxk = dG4dxk.subs(xj**2 + yj**2 + zj**2, Rj2)
dG4dxk = dG4dxk.subs(xk**2 + yk**2 + zk**2, Rk2)
dG4dxk = dG4dxk.subs((xj-xk)**2 + (yj-yk)**2 + (zj-zk)**2, Rjk2)
dG4dxk = dG4dxk.subs(sqrt(Rj2), Rj)
dG4dxk = dG4dxk.subs(sqrt(Rk2), Rk)
dG4dxk = dG4dxk.subs(sqrt(Rjk2), Rjk)
dG4dxk = dG4dxk.subs(xj*xk + yj*yk + zj*zk, RjDotRk)
dG4dxk = dG4dxk.subs(RjDotRk/(Rj*Rk), CosTheta)
dG4dxk = dG4dxk.subs(0.5*cos(pi*Rj/Rc) + 0.5, Fcj)
dG4dxk = dG4dxk.subs(0.5*cos(pi*Rk/Rc) + 0.5, Fck)
dG4dxk = dG4dxk.subs(0.5*cos(pi*Rjk/Rc) + 0.5, Fcjk)
dG4dxk = dG4dxk.subs(sin(pi*Rk/Rc)/(Rc), -2*dFck/pi)
dG4dxk = dG4dxk.subs(sin(pi*Rjk/Rc)/(Rc), -2*dFcjk/pi)
#print simplify(dG4dxk)

print codegen(("dG4dxj", simplify(dG4dxj)), "C", "file")[0][1]
print codegen(("dG4dxk", simplify(dG4dxk)), "C", "file")[0][1]