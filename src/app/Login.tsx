"use client";
import {useSession, signIn, signOut} from "next-auth/react";
import App from "./page";
export default function Login() {
    const { data: session } = useSession();
    if (session) {
        return (
          <>
            <button onClick={() => signOut()} type="button" className="btn btn-primary w-1/6 bg-black-500 text-white text-lg py-1 px-4">
              Sign Out
            </button>
          </>
        );
      } else {
        return (
          <>
            <button onClick={() => signIn()} type="button" className="btn btn-primary w-1/6 bg-black-500 text-white text-lg py-1 px-4">
              Sign In
            </button>
          </>
        );
      }
      
}

