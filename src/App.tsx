import { NavLink, Outlet } from 'react-router-dom';

const LINKS = [
  { to: '/', label: 'Predict', end: true },
  { to: '/batch', label: 'Batch' },
  { to: '/performance', label: 'Performance' },
  { to: '/history', label: 'Log' },
];

export default function App() {
  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="brand-name">GenoHybrid&#8209;VEP</span>
            <span className="brand-sub">Variant effect prediction</span>
          </div>
          <nav className="nav">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end}>
                {l.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <div className="page">
        <Outlet />
      </div>
    </>
  );
}
