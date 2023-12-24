import Image from 'next/image';
import { signIn, signOut } from "next-auth/react";
import githubLogo from ".//../../public/github-logo.png"

interface HeaderProps {
    session: any;
    selectedLanguage: string;
    setSelectedLanguage: (value: string) => void;
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
            <div className="w-full bg-slate-900 text-white text-lg py-2 px-4">
                Accelerated Computing Online
            </div>

            {!session?.user?.name ? (
                <button
                    onClick={() => signIn()}
                    type="button"
                    className="btn btn-primary bg-emerald-500 text-white w-5/12 text-base"
                >
                    Sign In
                </button>
            ) : (
                <div className="flex flex-row w-5/12">
                    <select
                        className="bg-blue-500 text-center text-white py-1 px-4 w-5/12 text-base "
                        onChange={(e) => setSelectedLanguage(e.target.value)}
                        value={selectedLanguage}
                    >
                        <option value="triton">Triton/Numba</option>
                        <option value="cuda">CUDA</option>
                    </select>
                    <button
                        className="bg-emerald-500 text-white w-1/3 py-1 px-4 text-base"
                        onClick={sendFunction}
                    >
                        Run Kernel
                    </button>
                    <button
                        onClick={() => signOut()}
                        type="button"
                        className="bg-red-500 btn btn-primary text-white w-1/3 py-1 px-4 text-base "
                    >
                        Sign Out
                    </button>
                </div>
            )}
            <div className="bg-gray-950">
                <a href="https://github.com/tfsingh/aconline">
                    <Image src={githubLogo} height={53} width={53} alt="github" />
                </a>
            </div>

        </div>
    );
};

export default Header;
