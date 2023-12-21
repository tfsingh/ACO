import Image from 'next/image';
import { signIn, signOut } from "next-auth/react";
import githubLogo from ".//../../public/github-logo.png"

interface HeaderProps {
    session: any;
    selectedLanguage: string;
    setSelectedLanguage: React.Dispatch<React.SetStateAction<string>>;
    sendFunction: () => void;
}

const Header: React.FC<HeaderProps> = ({
    session,
    selectedLanguage,
    setSelectedLanguage,
    sendFunction,
}) => {

    return (
        <div className="flex flex-row">
            <div className="w-full bg-slate-900 text-white text-lg py-1 px-4">
                Accelerated Computing, Online
            </div>
            <div className="bg-slate-900">
                <a href="https://github.com/tfsingh/aconline">
                    <Image src={githubLogo} height={50} width={50} alt="github" />
                </a>
            </div>
            {!session?.user?.name ? (
                <button
                    onClick={() => signIn()}
                    type="button"
                    className="border-2 border-emerald-500 btn btn-primary bg-slate-900 text-white py-1 px-4 w-5/12 text-base rounded"
                >
                    Sign In
                </button>
            ) : (
                <div className="flex flex-row w-5/12">
                    <select
                        className="border-2 border-blue-500 text-center text-white py-1 px-4 w-5/12 text-base bg-slate-900 rounded-l"
                        onChange={(e) => setSelectedLanguage(e.target.value)}
                        value={selectedLanguage}
                    >
                        <option value="triton">Triton/Numba</option>
                        <option value="cuda">CUDA</option>
                    </select>
                    <button
                        className="border-2 border-emerald-500 bg-slate-900 text-white w-1/3 py-1 px-4 text-base"
                        onClick={sendFunction}
                    >
                        Run Kernel
                    </button>
                    <button
                        onClick={() => signOut()}
                        type="button"
                        className="border-2 border-red-500 btn btn-primary text-white w-1/3 py-1 px-4 text-base bg-slate-900 rounded-r"
                    >
                        Sign Out
                    </button>
                </div>
            )}
        </div>
    );
};

export default Header;
