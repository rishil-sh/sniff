import type { Metadata } from "next";

import { History } from "@/components/check/history";

export const metadata: Metadata = { title: "History" };

const HistoryPage = () => <History />;

export default HistoryPage;
