import Link from 'next/link';

export default function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-charcoal text-white p-4 space-y-4">
      <div className="text-2xl font-bold mb-6">Navigation</div>
      <nav>
        <ul>
          <li>
            <Link href="/" className="flex items-center p-2 text-base font-normal text-white rounded-lg hover:bg-subtle-gray-dark group">
              Home
            </Link>
          </li>
          <li>
            <Link href="/dashboard" className="flex items-center p-2 text-base font-normal text-white rounded-lg hover:bg-subtle-gray-dark group">
              Student Data
            </Link>
          </li>
          <li>
            <Link href="/model-stats" className="flex items-center p-2 text-base font-normal text-white rounded-lg hover:bg-subtle-gray-dark group">
              Model Stats
            </Link>
          </li>
          <li>
            <Link href="/simulate" className="flex items-center p-2 text-base font-normal text-white rounded-lg hover:bg-subtle-gray-dark group">
              Simulate
            </Link>
          </li>
        </ul>
      </nav>
    </aside>
  );
}